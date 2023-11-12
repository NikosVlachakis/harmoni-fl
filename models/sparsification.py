import flwr as fl
import tensorflow as tf
import logging
import numpy as np
import sparse
import pickle
from utils.simple_utils import flatten_weights

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module


class Sparsifier:
    
    def __init__(self, method="weight_magnitude", sparsity_threshold=0.6, percentile=95):
        self.method = method
        self.sparsity_threshold = sparsity_threshold
        self.percentile = percentile

    def update_threshold_weight_based(self, weights):
        if self.method == "weight_magnitude":
            # Flatten all the weights
            all_weights = flatten_weights(weights)
            # Calculates  calculates the value of threshold below which percentile% of the absolute values of the weights in the model fall.
            self.sparsity_threshold = np.percentile(np.abs(all_weights), self.percentile)
            logger.info("Sparsity threshold is: %s", self.sparsity_threshold)
        else:
            self.sparsity_threshold = 0.7
   
    def serialize_sparse_coo(self,coo_matrix):
        data = coo_matrix.data.tolist()  # Convert data to list
        coords = coo_matrix.coords.tolist()  # Convert coords to list
        shape = list(coo_matrix.shape)  # Convert shape to list

        # Create a dictionary with all components
        serialized_data = {
            "data": data,
            "coords": coords,
            "shape": shape
        }

        # Serialize the dictionary using pickle or a similar library
        return pickle.dumps(serialized_data)

    def sparsify_weight(self,weight):
            """Convert a weight to sparse format using a sparsity threshold."""
            mask = np.abs(weight) > self.sparsity_threshold
            weight_masked = np.where(mask, weight, 0)  # Apply mask
            return sparse.COO.from_numpy(weight_masked)  # Convert to sparse COO format


    def sparsify_and_serialize_weights(self,weights):
        
        self.update_threshold_weight_based(weights)
        
        total_nnz = 0  # Initialize total non-zero element count
        serialized_sparse_weights = []
        for weight in weights:
            sparse_weight = self.sparsify_weight(weight)
            serialized_sparse_weight = self.serialize_sparse_coo(sparse_weight)
            serialized_sparse_weights.append(serialized_sparse_weight)

            # Log nnz for the current sparse weight and accumulate total nnz
            current_nnz = sparse_weight.nnz
            total_nnz += current_nnz

        return serialized_sparse_weights, total_nnz