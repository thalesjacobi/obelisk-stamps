from __future__ import annotations

import os
import argparse
from typing import Dict, List, Tuple

import mysql.connector
import tensorflow as tf
from dotenv import load_dotenv

from ml.model_utils import build_embedding_model


def load_stamp_rows_from_mysql(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str = "postbeeld_stamps",
    require_image_on_disk: bool = True,
) -> List[dict]:
    conn = mysql.connector.connect(
        host=host, port=port, user=user, password=password, database=database
    )
    cur = conn.cursor(dictionary=True)
    cur.execute(f"""
        SELECT id, image_path, title, source_url, price_value, price_currency, country, year, condition_text
        FROM {table}
        WHERE image_path IS NOT NULL AND image_path <> ''
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    out = []
    for r in rows:
        p = r.get("image_path")
        if not p:
            continue

        # Resolve relative paths if needed
        if not os.path.isabs(p):
            p = os.path.abspath(p)
            r["image_path"] = p

        if require_image_on_disk and not os.path.exists(p):
            continue

        if r.get("price_value") is None or not r.get("price_currency"):
            continue

        out.append(r)

    return out


def make_stamp_tf_dataset(
    rows: List[dict],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Dataset yields: ((view1, view2), label)
    Where label is unique per DB row (baseline retrieval approach).
    """
    db_ids = [int(r["id"]) for r in rows]
    id_to_class = {db_id: idx for idx, db_id in enumerate(db_ids)}

    paths = [r["image_path"] for r in rows]
    labels = [id_to_class[int(r["id"])] for r in rows]

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(rows), 5000), reshuffle_each_iteration=True)

    def _read_image(path: tf.Tensor) -> tf.Tensor:
        data = tf.io.read_file(path)
        img = tf.image.decode_jpeg(data, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, image_size, antialias=True)
        return img

    def _augment(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.75, 1.25)

        crop_scale = tf.random.uniform([], 0.75, 1.0)
        h = tf.cast(tf.shape(img)[0], tf.float32)
        w = tf.cast(tf.shape(img)[1], tf.float32)
        ch = tf.cast(h * crop_scale, tf.int32)
        cw = tf.cast(w * crop_scale, tf.int32)
        img = tf.image.random_crop(img, size=[ch, cw, 3])
        img = tf.image.resize(img, image_size, antialias=True)

        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)

        noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
        img = tf.clip_by_value(img + noise, 0.0, 1.0)
        return img

    def _map(path: tf.Tensor, label: tf.Tensor):
        img = _read_image(path)
        v1 = _augment(img)
        v2 = _augment(img)
        return (v1, v2), label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def train_stamp_embedding_model(
    ds_train,
    image_size=(224, 224),
    embedding_dim=256,
    epochs=10,
    model_save_path="models/stamp_embed.keras",
    base_trainable=False,
    learning_rate=1e-4,
):
    """
    MVP-safe trainer that works even if ds_train yields:
      - x
      - (x, y)
      - (x1, x2)     (two augmented views)
      - (x, y, w, ..)

    We ALWAYS take the first component as the model input 'x' and ignore the rest.
    This prevents 'model expects 1 input but got 2 tensors' errors.
    """
    model = build_embedding_model(
        image_size=image_size,
        embedding_dim=embedding_dim,
        base_trainable=base_trainable,
    )

    def make_dummy_y(x):
        # y has shape [batch, embedding_dim]
        return tf.zeros((tf.shape(x)[0], embedding_dim), dtype=tf.float32)

    # Robust extractor: works whether dataset yields x or (x, ...)
    def to_xy(*args):
        # If dataset yields a single item, args=(x,)
        x = args[0]

        # If someone upstream wrapped x itself as a tuple/list (rare), unwrap first element
        if isinstance(x, (tuple, list)):
            x = x[0]

        return x, make_dummy_y(x)

    ds_xy = ds_train.map(to_xy, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    model.fit(ds_xy, epochs=epochs)

    model.save(model_save_path)
    print(f"[ML] Saved model to: {model_save_path}")

    return model


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--embedding", type=int, default=256)
    ap.add_argument("--out", type=str, default="models/stamp_embed.keras")
    args = ap.parse_args()

    rows = load_stamp_rows_from_mysql(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        table="postbeeld_stamps",
        require_image_on_disk=True,
    )

    if not rows:
        raise RuntimeError("No usable rows found (check DB + image_path on disk).")

    ds = make_stamp_tf_dataset(
        rows=rows,
        image_size=(args.img, args.img),
        batch_size=args.batch,
    )

    train_stamp_embedding_model(
        ds_train=ds,
        image_size=(args.img, args.img),
        embedding_dim=args.embedding,
        epochs=args.epochs,
        model_out_path=args.out,
    )


if __name__ == "__main__":
    main()
