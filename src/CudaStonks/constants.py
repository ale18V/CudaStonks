import os

TEST_RATIO = 0.3
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
learning_rate = 0.01
regularization_strength = 1e-5
epochs = 2000
layers_size = [64, 16]
