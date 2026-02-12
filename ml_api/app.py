"""
ML API Service for Stamp Recognition

This is an independent API service that provides:
1. Stamp detection (YOLO) - detect individual stamps in images
2. Stamp matching (EfficientNet embeddings) - find similar stamps

The service loads models from Google Cloud Storage at startup.
Memory requirement: 4Gi (TensorFlow + YOLO + embeddings index)
"""

import os
import io
import pickle
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from website

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
GCS_BUCKET = os.getenv("GCS_MODEL_BUCKET", "obelisk-stamps-models")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/models"))

# Model file paths (in GCS and local cache)
MODEL_FILES = {
    "embeddings_index": "ref_embeddings.npy",
    "embeddings_rows": "ref_rows.pkl",
    "yolo_weights": "stamp_detector.pt",
}

# SavedModel directory files (for embedding model)
SAVEDMODEL_DIR = "stamp_embed_savedmodel"
SAVEDMODEL_FILES = [
    "fingerprint.pb",
    "saved_model.pb",
    "variables/variables.data-00000-of-00001",
    "variables/variables.index",
]

# ------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------
embedding_model = None
ref_embeddings: Optional[np.ndarray] = None
ref_rows: Optional[List[dict]] = None
stamp_detector = None
ml_ready = False
ml_error: Optional[str] = None


# ------------------------------------------------------------
# GCS DOWNLOAD
# ------------------------------------------------------------
def download_from_gcs(bucket_name: str, source_blob: str, dest_path: Path) -> bool:
    """Download a file from Google Cloud Storage."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest_path))
        print(f"[GCS] Downloaded {source_blob} -> {dest_path}")
        return True
    except Exception as e:
        print(f"[GCS] Error downloading {source_blob}: {e}")
        return False


def ensure_models_downloaded() -> bool:
    """Download all model files from GCS if not already cached."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download regular model files
    for key, filename in MODEL_FILES.items():
        local_path = MODEL_DIR / filename
        if not local_path.exists():
            print(f"[ML] Downloading {filename} from GCS...")
            if not download_from_gcs(GCS_BUCKET, filename, local_path):
                return False
        else:
            print(f"[ML] Using cached {filename}")

    # Download SavedModel directory files
    savedmodel_local = MODEL_DIR / SAVEDMODEL_DIR
    for rel_path in SAVEDMODEL_FILES:
        local_path = savedmodel_local / rel_path
        gcs_path = f"{SAVEDMODEL_DIR}/{rel_path}"
        if not local_path.exists():
            print(f"[ML] Downloading {gcs_path} from GCS...")
            if not download_from_gcs(GCS_BUCKET, gcs_path, local_path):
                return False
        else:
            print(f"[ML] Using cached {gcs_path}")

    return True


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
def load_ml_assets() -> bool:
    """Load all ML models into memory."""
    global embedding_model, ref_embeddings, ref_rows, stamp_detector, ml_ready, ml_error

    ml_error = None
    ml_ready = False

    # Ensure models are downloaded
    if not ensure_models_downloaded():
        ml_error = "Failed to download models from GCS"
        return False

    # Load embedding model (SavedModel format)
    try:
        import tensorflow as tf

        model_path = MODEL_DIR / SAVEDMODEL_DIR
        embedding_model = tf.saved_model.load(str(model_path))
        print(f"[ML] Loaded embedding model from {model_path}")
    except Exception as e:
        ml_error = f"Failed to load embedding model: {e}"
        print(f"[ML] {ml_error}")
        return False

    # Load embeddings index
    try:
        index_path = MODEL_DIR / MODEL_FILES["embeddings_index"]
        ref_embeddings = np.load(str(index_path))
        print(f"[ML] Loaded embeddings index: {ref_embeddings.shape}")
    except Exception as e:
        ml_error = f"Failed to load embeddings index: {e}"
        print(f"[ML] {ml_error}")
        return False

    # Load rows metadata
    try:
        rows_path = MODEL_DIR / MODEL_FILES["embeddings_rows"]
        with open(rows_path, "rb") as f:
            ref_rows = pickle.load(f)
        print(f"[ML] Loaded {len(ref_rows)} stamp records")
    except Exception as e:
        ml_error = f"Failed to load rows metadata: {e}"
        print(f"[ML] {ml_error}")
        return False

    # Load YOLO detector
    try:
        from ultralytics import YOLO
        weights_path = MODEL_DIR / MODEL_FILES["yolo_weights"]
        stamp_detector = YOLO(str(weights_path))
        print(f"[ML] Loaded YOLO detector from {weights_path}")
    except Exception as e:
        print(f"[ML] Warning: YOLO detector not loaded: {e}")
        stamp_detector = None  # Non-fatal, detection will be skipped

    ml_ready = True
    print("[ML] All models loaded successfully!")
    return True


# ------------------------------------------------------------
# INFERENCE FUNCTIONS
# ------------------------------------------------------------
def detect_stamps(image_bytes: bytes, confidence: float = 0.3) -> List[dict]:
    """Detect stamps in an image using YOLO."""
    if stamp_detector is None:
        return []

    # Save to temp file (YOLO needs file path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        results = stamp_detector(tmp_path, conf=confidence, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Ensure correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0]),
                    "class": "stamp",
                })

        # Sort by position (top-to-bottom, left-to-right)
        detections.sort(key=lambda d: (d["bbox"][1] // 50, d["bbox"][0]))
        return detections
    finally:
        os.unlink(tmp_path)


def get_embedding(image_bytes: bytes) -> np.ndarray:
    """Get embedding vector for an image."""
    import tensorflow as tf

    # Decode image
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (224, 224), antialias=True)
    img = tf.expand_dims(img, axis=0)

    # Get embedding - SavedModel uses 'serve' signature
    serve_fn = embedding_model.signatures['serving_default']
    result = serve_fn(image=img)
    # Get the output tensor (key may vary, get first output)
    output_key = list(result.keys())[0]
    emb = result[output_key].numpy().astype("float32")[0]
    emb /= np.linalg.norm(emb) + 1e-12
    return emb


def find_similar_stamps(query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
    """Find most similar stamps in the index."""
    # Compute similarities
    similarities = ref_embeddings @ query_embedding

    # Get top-k indices
    top_indices = np.argsort(-similarities)[:top_k]

    results = []
    for idx in top_indices:
        row = ref_rows[int(idx)]
        results.append({
            "similarity": float(similarities[idx]),
            "id": int(row["id"]),
            "title": row.get("title"),
            "country": row.get("country"),
            "year": row.get("year"),
            "condition_text": row.get("condition_text"),
            "price_value": float(row["price_value"]) if row.get("price_value") else None,
            "price_currency": row.get("price_currency"),
            "source_url": row.get("source_url"),
            "image_path": row.get("image_path"),
        })

    return results


# ------------------------------------------------------------
# API ROUTES
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if ml_ready else "unhealthy",
        "ml_ready": ml_ready,
        "error": ml_error,
        "models_loaded": {
            "embedding_model": embedding_model is not None,
            "embeddings_index": ref_embeddings is not None,
            "yolo_detector": stamp_detector is not None,
        },
        "index_size": len(ref_rows) if ref_rows else 0,
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    Detect stamps in an uploaded image.

    Request: multipart/form-data with 'image' file
    Response: {"detections": [{"bbox": [x1,y1,x2,y2], "confidence": 0.95}, ...]}
    """
    if not ml_ready:
        return jsonify({"error": "ML models not loaded", "details": ml_error}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    confidence = float(request.form.get("confidence", 0.3))
    detections = detect_stamps(image_bytes, confidence)

    return jsonify({
        "detections": detections,
        "count": len(detections),
    })


@app.route("/match", methods=["POST"])
def match():
    """
    Find similar stamps for an uploaded image.

    Request: multipart/form-data with 'image' file
    Response: {"matches": [{"similarity": 0.95, "title": "...", ...}, ...]}
    """
    if not ml_ready:
        return jsonify({"error": "ML models not loaded", "details": ml_error}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    top_k = int(request.form.get("top_k", 5))

    try:
        embedding = get_embedding(image_bytes)
        matches = find_similar_stamps(embedding, top_k)
        return jsonify({"matches": matches})
    except Exception as e:
        return jsonify({"error": "Matching failed", "details": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Full prediction: detect stamps and find matches for each.

    This is the main endpoint that handles both single and multi-stamp images.

    Request: multipart/form-data with 'image' file
    Response:
        Single stamp: {"matches": [...]}
        Multi-stamp: {"multi_stamp": true, "stamps": [{"bbox": ..., "matches": [...]}, ...]}
    """
    if not ml_ready:
        return jsonify({"error": "ML models not loaded", "details": ml_error}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    top_k = int(request.form.get("top_k", 3))
    confidence = float(request.form.get("confidence", 0.3))

    # Detect stamps
    detections = detect_stamps(image_bytes, confidence)

    if len(detections) > 1:
        # Multi-stamp image: match each detected stamp
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        stamps_results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Crop stamp
            x1 = max(0, x1 - 2)
            y1 = max(0, y1 - 2)
            x2 = min(w, x2 + 2)
            y2 = min(h, y2 + 2)

            crop = img.crop((x1, y1, x2, y2))

            # Convert crop to bytes
            crop_buffer = io.BytesIO()
            crop.save(crop_buffer, format="JPEG", quality=95)
            crop_bytes = crop_buffer.getvalue()

            # Get matches for this crop
            try:
                embedding = get_embedding(crop_bytes)
                matches = find_similar_stamps(embedding, top_k)
            except Exception as e:
                matches = []

            stamps_results.append({
                "bbox": det["bbox"],
                "detection_confidence": det["confidence"],
                "matches": matches,
            })

        return jsonify({
            "multi_stamp": True,
            "stamp_count": len(detections),
            "stamps": stamps_results,
        })
    else:
        # Single stamp (or no detection): match the whole image
        try:
            embedding = get_embedding(image_bytes)
            matches = find_similar_stamps(embedding, top_k)
            return jsonify({"matches": matches})
        except Exception as e:
            return jsonify({"error": "Matching failed", "details": str(e)}), 500


# ------------------------------------------------------------
# STARTUP - Lazy loading to avoid worker timeout
# ------------------------------------------------------------
print("[ML API] Starting up (models will load on first request)...")

def ensure_ml_loaded():
    """Ensure models are loaded before processing requests."""
    global ml_ready
    if not ml_ready:
        print("[ML API] Loading models on first request...")
        load_ml_assets()
    return ml_ready

@app.before_request
def before_request():
    """Load models on first request if not already loaded."""
    # Skip loading for health check to allow container to start
    if request.endpoint == 'health':
        return
    ensure_ml_loaded()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
