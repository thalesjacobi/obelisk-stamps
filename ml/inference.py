from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf


def convert_currency(
    amount: float,
    from_ccy: str,
    to_ccy: str,
    fx_rates: Dict[str, float],
) -> float:
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()

    if from_ccy not in fx_rates or to_ccy not in fx_rates:
        raise ValueError(f"Missing FX rate for {from_ccy} or {to_ccy}")

    if from_ccy == "EUR":
        amount_eur = amount
    else:
        amount_eur = amount / fx_rates[from_ccy]

    if to_ccy == "EUR":
        return amount_eur
    return amount_eur * fx_rates[to_ccy]


def predict_stamp_value_from_image(
    model: tf.keras.Model,
    ref_embeddings: np.ndarray,
    ref_rows: List[dict],
    query_image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    top_k: int = 5,
    return_currency: str = "EUR",
    fx_rates_eur_base: Optional[Dict[str, float]] = None,
) -> List[dict]:
    if fx_rates_eur_base is None:
        fx_rates_eur_base = {"EUR": 1.0, "USD": 1.08, "GBP": 0.86}

    if ref_embeddings.shape[0] != len(ref_rows):
        raise ValueError("ref_embeddings and ref_rows length mismatch")

    data = tf.io.read_file(query_image_path)
    img = tf.image.decode_jpeg(data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, antialias=True)
    img = tf.expand_dims(img, axis=0)

    q = model(img, training=False).numpy().astype("float32")[0]
    q /= np.linalg.norm(q) + 1e-12

    sims = ref_embeddings @ q
    idx = np.argsort(-sims)[:top_k]

    results = []
    for i in idx:
        row = ref_rows[int(i)]
        pv = float(row["price_value"])
        pc = str(row["price_currency"]).upper()

        price_converted = convert_currency(
            amount=pv,
            from_ccy=pc,
            to_ccy=return_currency,
            fx_rates=fx_rates_eur_base,
        )

        results.append({
            "match_db_id": int(row["id"]),
            "similarity": float(sims[int(i)]),
            "title": row.get("title"),
            "source_url": row.get("source_url"),
            "country": row.get("country"),
            "year": row.get("year"),
            "condition_text": row.get("condition_text"),
            "price_original": {"value": pv, "currency": pc},
            "price_converted": {"value": round(price_converted, 2), "currency": return_currency.upper()},
            "image_path_ref": row.get("image_path"),
        })

    return results
