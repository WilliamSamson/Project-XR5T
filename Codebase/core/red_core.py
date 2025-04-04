import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
import numpy as np
from collections import deque

# ===== Meta-Learner =====
class MetaLearner(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super(MetaLearner, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # Predict: [No Change, Expand, Compress]

    def forward(self, sequence):
        _, (h_n, _) = self.lstm(sequence)
        out = self.fc(h_n.squeeze(0))
        return torch.softmax(out, dim=1)

# ===== Cognitive Models =====
class CognitiveCore(nn.Module):
    def __init__(self, input_size, base_hidden_size, output_size, dropout_p=0.5):
        super(CognitiveCore, self).__init__()
        self.fc1 = nn.Linear(input_size, base_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(base_hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DynamicCognitiveCore(nn.Module):
    def __init__(self, input_size, base_hidden_size, total_hidden_size, output_size, dropout_p=0.5):
        super(DynamicCognitiveCore, self).__init__()
        self.fc1 = nn.Linear(input_size, base_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_extra = nn.Linear(base_hidden_size, total_hidden_size)
        self.fc2 = nn.Linear(total_hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc_extra(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===== Meta-Cognitive Optimization Engine =====
class SelfOptimizationEngine:
    def __init__(self, model, input_size, base_hidden_size, output_size, learning_rate=0.001, weight_decay=1e-4,
                 max_extra_capacity=256):
        self.model = model
        self.input_size = input_size
        self.base_hidden_size = base_hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.extra_capacity = 0
        self.max_extra_capacity = max_extra_capacity

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

        # Meta-structures
        self.meta_learner = MetaLearner()
        self.meta_optimizer = optim.Adam(self.meta_learner.parameters(), lr=1e-4)

        self.knowledge_bank = deque(maxlen=20)  # Store past (val_loss, grad_norm, capacity, lr)
        self.patience = 3
        self.improvement_thresh = 0.005

    def train_on_batch(self, x, y):
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                output = self.model(x)
                loss = self.loss_fn(output, y)
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(dataloader)

    def monitor_gradient_norm(self):
        total_grad_norm = 0.0
        count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item()
                count += 1
        return total_grad_norm / count if count > 0 else 0.0

    def evaluate_meta_action(self):
        if len(self.knowledge_bank) < 5:
            return 0  # Not enough data

        sequence = torch.tensor(list(self.knowledge_bank), dtype=torch.float32).unsqueeze(0)
        prediction = self.meta_learner(sequence)
        action = torch.argmax(prediction, dim=1).item()
        return action  # 0: No Change, 1: Expand, 2: Compress (not implemented)

    def dynamic_architecture_adjustment(self):
        if self.extra_capacity >= self.max_extra_capacity:
            print("Maximum extra capacity reached.")
            return

        increment = 32
        new_extra_capacity = self.extra_capacity + increment
        new_total_hidden = self.base_hidden_size + new_extra_capacity
        old_model = self.model

        new_model = DynamicCognitiveCore(self.input_size, self.base_hidden_size, new_total_hidden, self.output_size)

        # Transfer fc1
        new_model.fc1.weight.data.copy_(old_model.fc1.weight.data)
        new_model.fc1.bias.data.copy_(old_model.fc1.bias.data)

        identity = torch.eye(self.base_hidden_size)
        new_weight = torch.zeros(new_total_hidden, self.base_hidden_size)
        new_weight[:self.base_hidden_size, :] = identity
        new_model.fc_extra.weight.data.copy_(new_weight)
        new_model.fc_extra.bias.data.zero_()

        old_total_hidden = self.base_hidden_size + self.extra_capacity if hasattr(old_model, 'fc_extra') else self.base_hidden_size
        new_model.fc2.weight.data[:, :old_total_hidden].copy_(old_model.fc2.weight.data)
        if new_total_hidden > old_total_hidden:
            new_model.fc2.weight.data[:, old_total_hidden:].zero_()
        new_model.fc2.bias.data.copy_(old_model.fc2.bias.data)

        self.model = new_model
        self.extra_capacity = new_extra_capacity
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def self_refine(self, epochs, train_loader, val_loader):
        best_val_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(epochs):
            start = time.time()
            epoch_loss = 0
            for x, y in train_loader:
                loss = self.train_on_batch(x, y)
                epoch_loss += loss
            avg_train_loss = epoch_loss / len(train_loader)
            val_loss = self.validate(val_loader)
            grad_norm = self.monitor_gradient_norm()
            epoch_time = time.time() - start

            # Log training metrics
            print(f"Epoch {epoch + 1}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}, GradNorm={grad_norm:.4f}, Time={epoch_time:.2f}s")

            # Log into knowledge bank
            self.knowledge_bank.append([
                val_loss,
                grad_norm,
                self.base_hidden_size + self.extra_capacity,
                self.learning_rate
            ])

            # Meta-decision
            if val_loss < best_val_loss - self.improvement_thresh:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= self.patience:
                action = self.evaluate_meta_action()
                if action == 1:
                    print("🔧 Meta-Learner Trigger: EXPANSION")
                    self.dynamic_architecture_adjustment()
                elif action == 2:
                    print("⚠️ Meta-Learner Trigger: COMPRESSION (future work)")
                else:
                    print("📉 Meta-Learner Trigger: NO CHANGE")
                epochs_since_improvement = 0


# ===== Advanced Dataset =====
class AdvancedSyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=500, output_size=2, noise_std=0.2, device="cpu"):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.noise_std = noise_std

        assert input_size >= 4, "Input size must be >= 4"

        self.inputs = torch.randn(num_samples, input_size)
        half = input_size // 2
        first_half, second_half = self.inputs[:, :half], self.inputs[:, half:]

        sine = torch.sin(first_half) + torch.exp(-first_half)
        quad = second_half ** 2 + torch.log1p(torch.abs(second_half))
        s_mean = sine.mean(dim=1, keepdim=True)
        q_mean = quad.mean(dim=1, keepdim=True)
        i1 = s_mean * q_mean
        i2 = s_mean ** 2 + q_mean ** 2

        noise = torch.randn(num_samples, output_size) * noise_std
        self.outputs = torch.cat([s_mean, q_mean], dim=1) + 0.5 * i1 + 0.3 * i2 + noise

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# ===== Main =====
def main():
    torch.manual_seed(42)
    input_size = 10000
    base_hidden_size = 64
    output_size = 2
    batch_size = 64
    num_epochs = 100

    initial_model = CognitiveCore(input_size, base_hidden_size, output_size)
    engine = SelfOptimizationEngine(initial_model, input_size, base_hidden_size, output_size)

    dataset = AdvancedSyntheticDataset(num_samples=1000, input_size=input_size, output_size=output_size)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print("🚀 Initiating Meta-Learning Self Optimization")
    engine.self_refine(num_epochs, train_loader, val_loader)
    print("✅ Completed Training")

if __name__ == "__main__":
    main()
