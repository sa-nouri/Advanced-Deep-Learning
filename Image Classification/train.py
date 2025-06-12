"""
Training script for EfficientNet image classification.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import callbacks

from models.efficientnet import EfficientNetClassifier


def setup_logging(log_dir: Path) -> None:
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def load_dataset(
    dataset_name: str,
    batch_size: int,
    img_size: Tuple[int, int]
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """
    Load and preprocess the dataset.

    Args:
        dataset_name: Name of the dataset to load
        batch_size: Batch size for training
        img_size: Target image size (height, width)

    Returns:
        Tuple of (train_dataset, test_dataset, num_classes)
    """
    # Load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=["train", "test"],
        with_info=True,
        as_supervised=True
    )
    num_classes = ds_info.features["label"].num_classes

    # Preprocess images
    def preprocess(image, label):
        image = tf.image.resize(image, img_size)
        label = tf.one_hot(label, num_classes)
        return image, label

    # Configure datasets
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(preprocess)
    ds_test = ds_test.batch(batch_size, drop_remainder=True)

    return ds_train, ds_test, num_classes


def get_callbacks(
    checkpoint_dir: Path,
    log_dir: Path,
    patience: int = 5
) -> list:
    """
    Create training callbacks.

    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
        patience: Number of epochs to wait before early stopping

    Returns:
        List of callbacks
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'model_{epoch:02d}.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1
        )
    ]


def train(
    dataset_name: str,
    batch_size: int,
    epochs: int,
    img_size: Tuple[int, int],
    learning_rate: float,
    dropout_rate: float,
    output_dir: Path,
    weights: Optional[str] = None
) -> None:
    """
    Train the EfficientNet model.

    Args:
        dataset_name: Name of the dataset to use
        batch_size: Batch size for training
        epochs: Number of epochs to train
        img_size: Target image size (height, width)
        learning_rate: Initial learning rate
        dropout_rate: Dropout rate for regularization
        output_dir: Directory to save outputs
        weights: Path to pre-trained weights (optional)
    """
    # Setup
    setup_logging(output_dir / 'logs')
    logger = logging.getLogger(__name__)

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    ds_train, ds_test, num_classes = load_dataset(
        dataset_name,
        batch_size,
        img_size
    )

    # Create model
    logger.info("Creating model")
    model = EfficientNetClassifier(
        input_shape=(*img_size, 3),
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weights=weights
    )

    # Get callbacks
    callbacks_list = get_callbacks(
        output_dir / 'checkpoints',
        output_dir / 'logs'
    )

    # Train model
    logger.info("Starting training")
    history = model.train(
        ds_train,
        ds_test,
        epochs,
        callbacks=callbacks_list
    )

    # Save final model
    logger.info("Saving final model")
    model.save(str(output_dir / 'final_model.h5'))

    # Evaluate model
    logger.info("Evaluating model")
    metrics = model.evaluate(ds_test)
    logger.info(f"Test metrics: {metrics}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train EfficientNet model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cars196",
        help="Dataset name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        help="Number of epochs"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Learning rate"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.2,
        help="Dropout rate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to pre-trained weights"
    )

    args = parser.parse_args()

    train(
        args.dataset,
        args.batch_size,
        args.epochs,
        (args.img_size, args.img_size),
        args.learning_rate,
        args.dropout_rate,
        Path(args.output_dir),
        args.weights
    )


if __name__ == "__main__":
    main() 
