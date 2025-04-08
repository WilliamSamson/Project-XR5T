import numpy as np
import torch

# Generate random data for training (e.g., 100 samples, 10 features)
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)

# Convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Split data into training and validation
train_data = (X_train_tensor[:80], y_train_tensor[:80])  # 80% for training
val_data = (X_train_tensor[80:], y_train_tensor[80:])    # 20% for validation

training_data = train_data
validation_data = val_data
