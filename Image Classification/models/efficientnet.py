"""
EfficientNet model implementation for image classification.
"""
from typing import List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models, Model, optimizers, metrics
from tensorflow.keras.applications import EfficientNetB0


class EfficientNetClassifier:
    """EfficientNet-based image classifier with transfer learning support."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 196,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-2,
        weights: str = "imagenet",
    ):
        """
        Initialize the EfficientNet classifier.

        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            learning_rate: Initial learning rate
            weights: Pre-trained weights to use ('imagenet' or path to weights file)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weights = weights
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """Build the EfficientNet model architecture."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation
        x = self._get_augmentation_model()(inputs)
        
        # Base model
        base_model = EfficientNetB0(
            include_top=False,
            input_tensor=x,
            weights=self.weights
        )
        base_model.trainable = False

        # Classification head
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)

        # Create model
        model = Model(inputs, outputs, name="EfficientNet")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=self._get_metrics()
        )
        return model

    def _get_augmentation_model(self) -> Model:
        """Create data augmentation model."""
        return models.Sequential([
            layers.experimental.preprocessing.RandomRotation(factor=0.15),
            layers.experimental.preprocessing.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1
            ),
            layers.experimental.preprocessing.RandomFlip(),
            layers.experimental.preprocessing.RandomContrast(factor=0.1),
        ], name="img_augmentation")

    def _get_metrics(self) -> List[metrics.Metric]:
        """Get list of metrics to track during training."""
        return [
            metrics.FalseNegatives(name="fn"),
            metrics.FalsePositives(name="fp"),
            metrics.TrueNegatives(name="tn"),
            metrics.TruePositives(name="tp"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.Accuracy(name='acc'),
            metrics.AUC(name='auc'),
            metrics.AUC(name='prc', curve='PR')
        ]

    def unfreeze_model(self) -> None:
        """Unfreeze the base model for fine-tuning."""
        for layer in self.model.layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=self._get_metrics()
        )

    def train(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        epochs: int,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of epochs to train
            callbacks: List of callbacks to use during training

        Returns:
            Training history
        """
        return self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=callbacks,
            verbose=2
        )

    def evaluate(self, test_dataset: tf.data.Dataset) -> List[float]:
        """
        Evaluate the model on test data.

        Args:
            test_dataset: Test dataset

        Returns:
            List of evaluation metrics
        """
        return self.model.evaluate(test_dataset, verbose=2)

    def predict(self, images: tf.Tensor) -> tf.Tensor:
        """
        Make predictions on input images.

        Args:
            images: Input images tensor

        Returns:
            Predicted class probabilities
        """
        return self.model.predict(images)

    def save(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> 'EfficientNetClassifier':
        """
        Load a saved model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded EfficientNetClassifier instance
        """
        model = models.load_model(filepath)
        instance = cls()
        instance.model = model
        return instance 
