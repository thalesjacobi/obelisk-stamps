import os
import pickle
import numpy as np
import tensorflow as tf

from ml.train_embeddings import (
    load_stamp_rows_from_mysql,
    make_stamp_tf_dataset,
    train_stamp_embedding_model,
)
from ml.build_index import build_embedding_index

def main():
    # ----------------------------
    # Load .env from project root
    # ----------------------------
    from dotenv import load_dotenv
    load_dotenv()

    # ----------------------------
    # Config (from env, like app.py)
    # ----------------------------
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", "3306"))
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")

    table = os.getenv("STAMP_TABLE", "postbeeld_stamps")

    model_path = os.getenv("STAMP_MODEL_PATH", "models/stamp_embed.keras")
    index_path = os.getenv("STAMP_INDEX_PATH", "indexes/ref_embeddings.npy")
    rows_path = os.getenv("STAMP_ROWS_PATH", "indexes/ref_rows.pkl")

    image_size = (224, 224)
    batch_size = int(os.getenv("STAMP_BATCH_SIZE", "32"))
    epochs = int(os.getenv("STAMP_EPOCHS", "10"))
    embedding_dim = int(os.getenv("STAMP_EMBED_DIM", "256"))

    # Optional: limit rows for a smoke test (set STAMP_MAX_ROWS=2000 in .env)
    max_rows = os.getenv("STAMP_MAX_ROWS")
    max_rows = int(max_rows) if max_rows else None

    # ----------------------------
    # Validate env
    # ----------------------------
    missing = [k for k, v in {
        "DB_HOST": db_host,
        "DB_USER": db_user,
        "DB_PASSWORD": db_password,
        "DB_NAME": db_name,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {missing}. "
            f"Add them to your .env file in the project root."
        )

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(rows_path) or ".", exist_ok=True)

    # ----------------------------
    # Load rows
    # ----------------------------
    rows = load_stamp_rows_from_mysql(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
        table=table,
        require_image_on_disk=True,
    )
    if not rows:
        raise RuntimeError(
            "No training rows found. Check that postbeeld_stamps.image_path is set "
            "and the files exist on disk."
        )

    print(f"[ML] Training rows (before limit): {len(rows)}")

    if max_rows is not None:
        rows = rows[:max_rows]
        print(f"[ML] Using subset for this run (STAMP_MAX_ROWS): {len(rows)}")

    # ----------------------------
    # Build dataset
    # make_stamp_tf_dataset() in your repo returns a SINGLE object (likely tf.data.Dataset)
    # but we also support tuple returns if you later change it.
    # ----------------------------
    ds_result = make_stamp_tf_dataset(
        rows=rows,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    ds_train = None

    # Case 1: single dataset returned
    if isinstance(ds_result, tf.data.Dataset):
        ds_train = ds_result
        print("[ML] make_stamp_tf_dataset returned a single tf.data.Dataset")

    # Case 2: tuple/list returned
    elif isinstance(ds_result, (tuple, list)):
        print(f"[ML] make_stamp_tf_dataset returned a tuple/list of length {len(ds_result)}")
        if len(ds_result) == 0:
            raise ValueError("make_stamp_tf_dataset returned an empty tuple/list.")

        # Most likely patterns:
        #  - (ds_train, id_to_class, class_to_row)
        #  - (ds_train, ds_val, id_to_class, class_to_row)
        ds_train = ds_result[0]

        if not isinstance(ds_train, tf.data.Dataset):
            raise TypeError(
                "First item returned by make_stamp_tf_dataset is not a tf.data.Dataset. "
                f"Got: {type(ds_train)}"
            )

    else:
        raise TypeError(
            "make_stamp_tf_dataset returned an unsupported type. "
            f"Got: {type(ds_result)}. Expected tf.data.Dataset or tuple/list."
        )

    # ----------------------------
    # Train model (saves to model_path)
    # ----------------------------
    model = train_stamp_embedding_model(
        ds_train=ds_train,
        image_size=image_size,
        embedding_dim=embedding_dim,
        epochs=epochs,
        model_save_path=model_path,
    )

    # ----------------------------
    # Build index + save
    # ----------------------------
    ref_embeddings, ref_rows = build_embedding_index(
        model=model,
        rows=rows,
        image_size=image_size,
        batch_size=64,
    )

    np.save(index_path, ref_embeddings)
    with open(rows_path, "wb") as f:
        pickle.dump(ref_rows, f)

    print(f"[ML] Saved index to: {index_path}  shape={ref_embeddings.shape}")
    print(f"[ML] Saved rows  to: {rows_path}  count={len(ref_rows)}")
    print("[ML] Done.")


if __name__ == "__main__":
    main()
