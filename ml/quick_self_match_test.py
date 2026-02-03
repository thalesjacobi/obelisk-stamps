import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "models/stamp_embed.keras"
EMB_PATH = "indexes/ref_embeddings.npy"
ROWS_PATH = "indexes/ref_rows.pkl"

def main():
    # IMPORTANT:
    # Import model_utils first so the @register_keras_serializable decorator runs
    # and Keras can locate the custom layer (L2Normalize) during model loading.
    import ml.model_utils  # noqa: F401

    model = keras.models.load_model(MODEL_PATH)

    emb = np.load(EMB_PATH)
    with open(ROWS_PATH, "rb") as f:
        rows = pickle.load(f)

    if len(rows) == 0:
        raise RuntimeError("ref_rows.pkl is empty")

    row0 = rows[0]
    img_path = row0.get("image_path") or row0.get("path") or row0.get("image")
    if not img_path:
        raise RuntimeError("Could not find image_path/path/image key in first row")

    print("Model:", MODEL_PATH)
    print("Embeddings:", emb.shape)
    print("Rows:", len(rows))
    print("Test image:", img_path)

    raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)

    q = model(img, training=False).numpy()[0]
    scores = emb @ q
    top = int(np.argmax(scores))

    top_row = rows[top]
    top_img_path = top_row.get("image_path") or top_row.get("path") or top_row.get("image")

    print("Top index:", top)
    print("Top score:", float(scores[top]))
    print("Top image:", top_img_path)


if __name__ == "__main__":
    main()
