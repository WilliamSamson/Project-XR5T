import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. Cognitive Core Definition
# -------------------------------
class CognitiveCore(nn.Module):
    def __init__(self, input_size, base_hidden_size, output_size):
        super(CognitiveCore, self).__init__()
        # fc1 outputs the base hidden representation.
        self.fc1 = nn.Linear(input_size, base_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(base_hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 1a. Dynamic Cognitive Core Definition
# -------------------------------
# This architecture includes an extra hidden layer to expand capacity.
class DynamicCognitiveCore(nn.Module):
    def __init__(self, input_size, base_hidden_size, total_hidden_size, output_size):
        """
        Args:
            base_hidden_size (int): The fixed size output from fc1.
            total_hidden_size (int): base_hidden_size + extra_capacity.
        """
        super(DynamicCognitiveCore, self).__init__()
        self.fc1 = nn.Linear(input_size, base_hidden_size)
        self.relu = nn.ReLU()
        # fc_extra maps from base_hidden_size to total_hidden_size.
        self.fc_extra = nn.Linear(base_hidden_size, total_hidden_size)
        self.fc2 = nn.Linear(total_hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc_extra(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 2. Self-Optimization Engine with Weight Transfer
# -------------------------------
class SelfOptimizationEngine:
    def __init__(self, model, input_size, base_hidden_size, output_size, learning_rate=0.001):
        self.model = model
        self.input_size = input_size
        self.base_hidden_size = base_hidden_size  # fixed dimension from fc1
        self.output_size = output_size
        self.learning_rate = learning_rate
        # extra_capacity tracks how much we've expanded (initially 0)
        self.extra_capacity = 0
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
        # Increase extra capacity by a fixed amount (e.g., 16 units)
        increment = 16
        new_extra_capacity = self.extra_capacity + increment
        new_total_hidden = self.base_hidden_size + new_extra_capacity
        print(f"Dynamic Architecture Adjustment Triggered: Increasing total hidden capacity from {self.base_hidden_size + self.extra_capacity} to {new_total_hidden}.")

        # Save the old model for weight transfer.
        old_model = self.model

        # Instantiate new model with dynamic architecture.
        new_model = DynamicCognitiveCore(self.input_size, self.base_hidden_size, new_total_hidden, self.output_size)

        # --- Weight Transfer ---
        # 1. Transfer fc1 weights (dimensions: [base_hidden_size, input_size])
        new_model.fc1.weight.data.copy_(old_model.fc1.weight.data)
        new_model.fc1.bias.data.copy_(old_model.fc1.bias.data)

        # 2. Initialize fc_extra to act as identity on the base_hidden_size part.
        # fc_extra weight shape: [new_total_hidden, base_hidden_size]
        # Create an identity for the first base_hidden_size rows.
        identity = torch.eye(self.base_hidden_size, device=new_model.fc_extra.weight.device)
        new_weight = torch.zeros(new_total_hidden, self.base_hidden_size, device=new_model.fc_extra.weight.device)
        new_weight[:self.base_hidden_size, :] = identity
        new_model.fc_extra.weight.data.copy_(new_weight)
        new_model.fc_extra.bias.data.zero_()

        # 3. Transfer fc2 weights:
        # If the old model was a CognitiveCore, its fc2 weight shape is [output_size, base_hidden_size].
        # If it was DynamicCognitiveCore, its fc2 weight shape is [output_size, (base_hidden_size + old_extra_capacity)].
        if hasattr(old_model, 'fc_extra'):
            old_total_hidden = self.base_hidden_size + self.extra_capacity
        else:
            old_total_hidden = self.base_hidden_size
        # Copy old fc2 weights into the first old_total_hidden columns.
        new_model.fc2.weight.data[:, :old_total_hidden].copy_(old_model.fc2.weight.data)
        # Initialize the new columns (if any) to zero.
        if new_total_hidden > old_total_hidden:
            new_model.fc2.weight.data[:, old_total_hidden:].zero_()
        new_model.fc2.bias.data.copy_(old_model.fc2.bias.data)

        # Update the engine's model and optimizer.
        self.model = new_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Update extra_capacity for future adjustments.
        self.extra_capacity = new_extra_capacity

    def self_refine(self, epochs, dataloader):
        # Adaptive Learning Rate scheduler: Reduce LR on plateau.
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
# 3. Advanced Synthetic Dataset Definition
# -------------------------------
class AdvancedSyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=500, output_size=2, noise_std=0.2, device="cpu"):
        """
        A synthetic dataset with multi-modal feature transformations.
        - First half of input: Sinusoidal + Exponential components.
        - Second half: Quadratic + Logarithmic components.
        - Interaction terms: Multiplicative and cross-feature interactions.
        - Noise: Gaussian perturbation for stochasticity.
        """
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.device = device

        # Ensure input size is valid.
        assert input_size >= 4, "Input size must be at least 4 for meaningful feature splits."

        # Generate random input data.
        self.inputs = torch.randn(num_samples, input_size, device=device)

        # Split inputs into two feature sets.
        half = input_size // 2
        first_half, second_half = self.inputs[:, :half], self.inputs[:, half:]

        # Apply transformations.
        sine_component = torch.sin(first_half) + torch.exp(-first_half)
        quadratic_component = second_half ** 2 + torch.log1p(torch.abs(second_half))

        # Feature aggregations.
        sine_feature = sine_component.mean(dim=1, keepdim=True)
        quadratic_feature = quadratic_component.mean(dim=1, keepdim=True)

        # Interaction terms.
        interaction_1 = sine_feature * quadratic_feature
        interaction_2 = sine_feature ** 2 + quadratic_feature ** 2

        # Construct final outputs.
        linear_output = torch.cat([sine_feature, quadratic_feature], dim=1)
        interaction_term = 0.5 * interaction_1 + 0.3 * interaction_2

        # Add controlled Gaussian noise.
        noise = torch.randn(num_samples, output_size, device=device) * noise_std
        self.outputs = linear_output + interaction_term + noise

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# -------------------------------
# 4. Main Training Script
# -------------------------------
def main():
    # For reproducibility.
    torch.manual_seed(42)

    # Hyperparameters.
    input_size = 500
    base_hidden_size = 32  # Fixed base dimension.
    output_size = 2
    learning_rate = 0.001
    num_epochs = 5000
    batch_size = 32

    # Instantiate the initial model (CognitiveCore) with base architecture.
    initial_model = CognitiveCore(input_size, base_hidden_size, output_size)
    # Create the engine with the initial model; extra_capacity is initially 0.
    engine = SelfOptimizationEngine(initial_model, input_size, base_hidden_size, output_size, learning_rate=learning_rate)
    # Use the advanced synthetic dataset.
    dataset = AdvancedSyntheticDataset(num_samples=1000, input_size=input_size, output_size=output_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training Loop.
    print("Starting training...")
    engine.self_refine(num_epochs, dataloader)
    print("Training complete.")

if __name__ == "__main__":
    main()
