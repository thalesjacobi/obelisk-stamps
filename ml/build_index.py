from __future__ import annotations

import os
import argparse
import pickle
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from ml.train_embeddings import load_stamp_rows_from_mysql


def build_embedding_index(
    model: tf.keras.Model,
    rows: List[dict],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[dict]]:
    paths = [r["image_path"] for r in rows]

    def _load(path):
        data = tf.io.read_file(path)
        img = tf.image.decode_jpeg(data, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, image_size, antialias=True)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    embs = []
    for batch in ds:
        z = model(batch, training=False).numpy().astype("float32")
        embs.append(z)

    ref_embeddings = np.concatenate(embs, axis=0)
    ref_embeddings /= np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-12
    return ref_embeddings, rows


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/stamp_embed.keras")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="indexes")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = tf.keras.models.load_model(args.model)

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

    ref_embeddings, ref_rows = build_embedding_index(
        model=model,
        rows=rows,
        image_size=(args.img, args.img),
        batch_size=args.batch,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, "ref_embeddings.npy")
    pkl_path = os.path.join(args.out_dir, "ref_rows.pkl")

    np.save(npy_path, ref_embeddings)
    with open(pkl_path, "wb") as f:
        pickle.dump(ref_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved embeddings: {npy_path}")
    print(f"Saved rows:       {pkl_path}")
    print(f"Total refs:       {len(ref_rows)}")


if __name__ == "__main__":
    main()
