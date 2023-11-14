from enum import Enum, auto

class Names(Enum):
    MAX_CPU_USAGE = auto()
    MAX_MEMORY_USAGE_PERCENTAGE = auto()
    INCOMING_BANDWIDTH = auto()
    OUTGOING_BANDWIDTH = auto()
    LEARNING_RATE_BASED_ON_INCOMING_BANDWIDTH = auto()
    EPOCH_ADJUSTMENT_BASED_ON_CPU_UTILIZATION = auto()
    CONTAINER_CPU_CORES = auto()
    CONTAINER_MEMORY_LIMIT = auto()
    ADAPTIVE_BATCH_SIZE_BASED_ON_MEMORY_UTILIZATION = auto()
    ADAPTIVE_DATA_SAMPLING_BASED_ON_MEMORY_UTILIZATION = auto()
    MODEL_LAYER_FREEZING_BASED_ON_HIGH_CPU_UTILIZATION = auto()
    FREEZE_LAYERS_PERCENTAGE = auto()
    GRADIENT_CLIPPING_BASED_ON_HIGH_CPU_UTILIZATION = auto()
    WEIGHT_PRECISION_BASED_ON_HIGH_CPU_UTILIZATION = auto()
    INCLUDE_CLIENTS_WITHIN_SPECIFIC_THRESHOLDS = auto()
    SPARSIFICATION_BASED_ON_OUTGOING_BANDWIDTH = auto()

    def __str__(self):
        """Return the string representation of the enum member."""
        return self.name.lower()

    @property
    def value(self):
        """Return the value of the enum member."""
        return str(self)
