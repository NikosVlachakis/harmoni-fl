from contextlib import contextmanager
import logging
import tensorflow as tf
from utils.names import Names

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)    

class ModelAdjuster:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.frozen_layers = []
        self.adjustment_mapper = {
            Names.FREEZE_LAYERS_PERCENTAGE.value: self.apply_freeze_layers_percentage,
            # Add other model-related configurations here
        }

    @contextmanager
    def apply_adjustments(self, config: dict):
        applied_adjustments = []
        for key, value in config.items():
            if key in self.adjustment_mapper:
                applied = self.adjustment_mapper[key](value)
                if applied:
                    applied_adjustments.append(key)
                
        try:
            yield
        finally:
            self.revert_adjustments(applied_adjustments)

    def revert_adjustments(self, applied_adjustments):
        for adjustment in applied_adjustments:
            revert_method = getattr(self, f"revert_{adjustment}", None)
            if revert_method:
                revert_method()

    def apply_freeze_layers_percentage(self, percentage: float):
        if percentage > 0:
            total_layers = len(self.model.layers)
            layers_to_freeze = int(total_layers * (percentage / 100))
            self.freeze_layers(layers_to_freeze)
            return True
        return False

    def freeze_layers(self, layers_to_freeze: int):
        for layer in self.model.layers[:layers_to_freeze]:
            layer.trainable = False
            self.frozen_layers.append(layer)
        logger.info(f"Freezed first {layers_to_freeze} layers.")

    def revert_freeze_layers_percentage(self):
        self.unfreeze_layers()

    def unfreeze_layers(self):
        for layer in self.frozen_layers:
            layer.trainable = True
