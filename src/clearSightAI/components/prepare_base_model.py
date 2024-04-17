from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Multiply
from tensorflow.keras.models import Model
from clearSightAI.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    def update_base_model(self):
        if self.model is None:
            raise ValueError("No custom model loaded. Please load a model first.")

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add attention mechanism
        attention_output = PrepareBaseModel.spatial_attention(model.get_layer('block3_conv3').output)


        flatten_in = tf.keras.layers.Flatten()(attention_output)

        dense1 = Dense(128, activation='relu')(flatten_in)  # Reduced number of units

        output = Dense(1, activation='sigmoid')(dense1)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(output)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )


        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size'),
            metrics=["accuracy"]
        )


        full_model.summary()
        return full_model

    @staticmethod
    def spatial_attention(input_feature):
        # Compute attention map
        attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(input_feature)
        # Multiply attention map with input feature
        weighted_feature = Multiply()([input_feature, attention])
        return weighted_feature

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
