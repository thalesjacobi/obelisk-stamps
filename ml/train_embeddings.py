"""
Contrastive Learning for Stamp Embeddings

This script trains an EfficientNetB0-based embedding model using triplet loss.
The model learns to produce similar embeddings for visually similar stamps
and different embeddings for dissimilar stamps.

Training approach:
1. Group stamps by country+year (similar stamps)
2. For each anchor stamp, find a positive (same group) and negative (different group)
3. Train with triplet loss: d(anchor, positive) < d(anchor, negative) + margin
"""

from __future__ import annotations

import os
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
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
    """Load stamp records from MySQL database."""
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


def group_stamps_by_similarity(rows: List[dict]) -> Dict[str, List[dict]]:
    """
    Group stamps by country+year for creating positive pairs.
    Stamps from the same country and year are likely to be similar.
    """
    groups = defaultdict(list)
    for r in rows:
        country = r.get("country", "unknown") or "unknown"
        year = r.get("year", "unknown") or "unknown"
        key = f"{country}_{year}"
        groups[key].append(r)
    return dict(groups)


def create_triplet_generator(
    rows: List[dict],
    groups: Dict[str, List[dict]],
    image_size: Tuple[int, int] = (224, 224),
):
    """
    Generator that yields (anchor, positive, negative) triplets.

    - Anchor: a random stamp
    - Positive: another stamp from the same country+year (similar)
    - Negative: a stamp from a different country+year (different)
    """
    # Build reverse lookup: row -> group_key
    row_to_group = {}
    for r in rows:
        country = r.get("country", "unknown") or "unknown"
        year = r.get("year", "unknown") or "unknown"
        row_to_group[r["id"]] = f"{country}_{year}"

    # Filter groups with at least 2 stamps (needed for positive pairs)
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    all_group_keys = list(valid_groups.keys())

    if len(all_group_keys) < 2:
        raise ValueError("Need at least 2 groups with 2+ stamps for triplet training")

    def read_and_preprocess(path: str) -> np.ndarray:
        """Read and preprocess a single image."""
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, image_size, antialias=True)
            return img.numpy()
        except Exception:
            # Return a blank image on error
            return np.zeros((*image_size, 3), dtype=np.float32)

    def augment(img: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        img = tf.convert_to_tensor(img)

        # Random flip
        if random.random() > 0.5:
            img = tf.image.flip_left_right(img)

        # Random brightness/contrast
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)

        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        img = tf.image.rot90(img, k=k)

        img = tf.clip_by_value(img, 0.0, 1.0)
        return img.numpy()

    while True:
        # Pick a random group for anchor and positive
        anchor_group_key = random.choice(all_group_keys)
        anchor_group = valid_groups[anchor_group_key]

        # Pick anchor and positive from same group
        if len(anchor_group) >= 2:
            anchor_row, positive_row = random.sample(anchor_group, 2)
        else:
            continue

        # Pick negative from a different group
        negative_group_keys = [k for k in all_group_keys if k != anchor_group_key]
        negative_group_key = random.choice(negative_group_keys)
        negative_row = random.choice(valid_groups[negative_group_key])

        # Load and augment images
        anchor_img = augment(read_and_preprocess(anchor_row["image_path"]))
        positive_img = augment(read_and_preprocess(positive_row["image_path"]))
        negative_img = augment(read_and_preprocess(negative_row["image_path"]))

        yield anchor_img, positive_img, negative_img


def make_triplet_dataset(
    rows: List[dict],
    groups: Dict[str, List[dict]],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset that yields triplets."""

    def generator():
        gen = create_triplet_generator(rows, groups, image_size)
        for triplet in gen:
            yield triplet

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
        )
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class TripletLoss(tf.keras.losses.Loss):
    """
    Triplet loss for contrastive learning.

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Encourages:
    - anchor and positive embeddings to be close
    - anchor and negative embeddings to be far apart
    """

    def __init__(self, margin: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        # y_pred is expected to be (anchor_emb, positive_emb, negative_emb) stacked
        # We'll handle this in the training loop instead
        return tf.constant(0.0)


class TripletModel(tf.keras.Model):
    """
    Wrapper model that computes triplet loss from anchor, positive, negative inputs.
    """

    def __init__(self, embedding_model: tf.keras.Model, margin: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        # For inference, just return embeddings
        return self.embedding_model(inputs, training=training)

    def train_step(self, data):
        anchor, positive, negative = data

        with tf.GradientTape() as tape:
            # Get embeddings for all three
            anchor_emb = self.embedding_model(anchor, training=True)
            positive_emb = self.embedding_model(positive, training=True)
            negative_emb = self.embedding_model(negative, training=True)

            # L2 normalize embeddings
            anchor_emb = tf.nn.l2_normalize(anchor_emb, axis=1)
            positive_emb = tf.nn.l2_normalize(positive_emb, axis=1)
            negative_emb = tf.nn.l2_normalize(negative_emb, axis=1)

            # Compute distances (squared L2)
            pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=1)

            # Triplet loss
            loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
            loss = tf.reduce_mean(loss)

        # Update weights
        gradients = tape.gradient(loss, self.embedding_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.embedding_model.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        anchor, positive, negative = data

        anchor_emb = self.embedding_model(anchor, training=False)
        positive_emb = self.embedding_model(positive, training=False)
        negative_emb = self.embedding_model(negative, training=False)

        anchor_emb = tf.nn.l2_normalize(anchor_emb, axis=1)
        positive_emb = tf.nn.l2_normalize(positive_emb, axis=1)
        negative_emb = tf.nn.l2_normalize(negative_emb, axis=1)

        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=1)

        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        loss = tf.reduce_mean(loss)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def train_stamp_embedding_model(
    rows: List[dict],
    image_size: Tuple[int, int] = (224, 224),
    embedding_dim: int = 256,
    epochs: int = 10,
    batch_size: int = 32,
    steps_per_epoch: int = 500,
    model_save_path: str = "models/stamp_embed.keras",
    base_trainable: bool = True,
    learning_rate: float = 1e-4,
    margin: float = 0.5,
):
    """
    Train embedding model with triplet loss.

    Args:
        rows: List of stamp records from database
        image_size: Input image size
        embedding_dim: Output embedding dimension
        epochs: Number of training epochs
        batch_size: Batch size for training
        steps_per_epoch: Number of batches per epoch
        model_save_path: Path to save the trained model
        base_trainable: Whether to fine-tune the EfficientNet base
        learning_rate: Learning rate for optimizer
        margin: Triplet loss margin
    """
    print(f"[ML] Building embedding model (embedding_dim={embedding_dim})...")

    # Build the base embedding model
    embedding_model = build_embedding_model(
        image_size=image_size,
        embedding_dim=embedding_dim,
        base_trainable=base_trainable,
    )

    # Wrap in triplet model
    triplet_model = TripletModel(embedding_model, margin=margin)

    triplet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )

    # Group stamps for positive pair mining
    print("[ML] Grouping stamps by country+year...")
    groups = group_stamps_by_similarity(rows)
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"[ML] Found {len(valid_groups)} groups with 2+ stamps")
    print(f"[ML] Total stamps in valid groups: {sum(len(v) for v in valid_groups.values())}")

    # Create dataset
    print("[ML] Creating triplet dataset...")
    ds = make_triplet_dataset(rows, groups, image_size, batch_size)

    # Train
    print(f"[ML] Training for {epochs} epochs, {steps_per_epoch} steps each...")
    print(f"[ML] Margin: {margin}, Learning rate: {learning_rate}")
    print("-" * 60)

    triplet_model.fit(
        ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Save the embedding model (not the triplet wrapper)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    embedding_model.save(model_save_path)
    print(f"[ML] Saved model to: {model_save_path}")

    return embedding_model


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Train stamp embedding model with contrastive learning")
    ap.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    ap.add_argument("--batch", type=int, default=32, help="Batch size")
    ap.add_argument("--steps", type=int, default=500, help="Steps per epoch")
    ap.add_argument("--img", type=int, default=224, help="Image size")
    ap.add_argument("--embedding", type=int, default=256, help="Embedding dimension")
    ap.add_argument("--margin", type=float, default=0.5, help="Triplet loss margin")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--finetune", action="store_true", help="Fine-tune EfficientNet base")
    ap.add_argument("--out", type=str, default="models/stamp_embed.keras", help="Output model path")
    args = ap.parse_args()

    print("[ML] Loading stamp data from MySQL...")
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

    print(f"[ML] Loaded {len(rows)} stamps from database")

    train_stamp_embedding_model(
        rows=rows,
        image_size=(args.img, args.img),
        embedding_dim=args.embedding,
        epochs=args.epochs,
        batch_size=args.batch,
        steps_per_epoch=args.steps,
        model_save_path=args.out,
        base_trainable=args.finetune,
        learning_rate=args.lr,
        margin=args.margin,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Next step: Build the search index:")
    print("  python -m ml.build_index")
    print("=" * 60)


if __name__ == "__main__":
    main()
