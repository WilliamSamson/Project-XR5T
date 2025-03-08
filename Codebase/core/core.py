import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. Cognitive Core Definition
# -------------------------------
class CognitiveCore(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CognitiveCore, self).__init__()
        # Initial architecture: two linear layers with a ReLU activation.
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 1a. Dynamic Cognitive Core Definition
# -------------------------------
# This architecture includes an extra hidden layer for increased capacity.
class DynamicCognitiveCore(nn.Module):
    def __init__(self, input_size, hidden_size, extra_hidden_size, output_size):
        super(DynamicCognitiveCore, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_extra = nn.Linear(hidden_size, extra_hidden_size)
        self.fc2 = nn.Linear(extra_hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc_extra(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 2. Self-Optimization Engine
# -------------------------------
class SelfOptimizationEngine:
    def __init__(self, model, input_size, hidden_size, output_size, learning_rate=0.01):
        self.model = model
        self.input_size = input_size
        self.hidden_size = hidden_size  # Current hidden layer size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()  # Placeholder loss; adjust as needed

    def train_on_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def dynamic_architecture_adjustment(self):
        # Increase hidden capacity by a fixed amount (e.g., 16 units)
        extra_hidden_size = self.hidden_size + 16
        print("Dynamic Architecture Adjustment Triggered: Reinitializing model with increased capacity.")
        # Instantiate new model with a dynamic architecture (extra hidden layer)
        new_model = DynamicCognitiveCore(self.input_size, self.hidden_size, extra_hidden_size, self.output_size)
        # Update engine's model and optimizer (weights are reinitialized)
        self.model = new_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Update hidden_size reference for future adjustments
        self.hidden_size = extra_hidden_size

    def self_refine(self, epochs, dataloader):
        # Adaptive Learning Rate scheduler: Reduce LR on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        best_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in dataloader:
                batch_loss = self.train_on_batch(x, y)
                epoch_loss += batch_loss

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}: Average Loss = {avg_loss:.4f}")

            # --- Meta-Learning Hook: Log Gradient Norms ---
            total_grad_norm = 0.0
            count = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item()
                    count += 1
            avg_grad_norm = total_grad_norm / count if count > 0 else 0
            print(f"  Average Gradient Norm: {avg_grad_norm:.4f}")

            # --- Adaptive Learning Rate Adjustment ---
            scheduler.step(avg_loss)

            # --- Dynamic Architecture Adjustment Trigger ---
            if avg_loss < best_loss - 0.01:  # threshold for improvement
                best_loss = avg_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= 2:
                self.dynamic_architecture_adjustment()
                epochs_since_improvement = 0  # Reset counter after adjustment

# -------------------------------
# 3. Synthetic Dataset Definition (Updated)
# -------------------------------
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=10, output_size=2):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size

        # Create random input data
        self.inputs = torch.randn(num_samples, input_size)
        # Define weights and bias for a non-linear transformation.
        self.weights = torch.randn(input_size, output_size)
        self.bias = torch.randn(output_size)
        # Non-linear relationship: using sine activation to generate outputs
        linear_combination = self.inputs.mm(self.weights) + self.bias
        # Increased noise and non-linearity make learning more challenging.
        self.outputs = torch.sin(linear_combination) + 0.2 * torch.randn(num_samples, output_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# -------------------------------
# 4. Main Training Script
# -------------------------------
def main():
    # For reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    input_size = 10
    hidden_size = 32
    output_size = 2
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32

    # Instantiate the initial model (CognitiveCore), engine, and dataset
    initial_model = CognitiveCore(input_size, hidden_size, output_size)
    engine = SelfOptimizationEngine(initial_model, input_size, hidden_size, output_size, learning_rate=learning_rate)
    dataset = SyntheticDataset(num_samples=1000, input_size=input_size, output_size=output_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training Loop
    print("Starting training...")
    engine.self_refine(num_epochs, dataloader)
    print("Training complete.")

if __name__ == "__main__":
    main()
