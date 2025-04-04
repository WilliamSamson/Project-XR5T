import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

# ChaosCore Engine - Double Pendulum Model with RK4
class ChaosEngine:
    def __init__(self, init_state, params):
        self.state = np.array(init_state, dtype=np.float32)
        self.params = np.array(params, dtype=np.float32)
        self.t = 0

    def dynamics(self, t, y):
        l1, l2, m1, m2, g = self.params
        θ1, ω1, θ2, ω2 = y

        dω1 = (-g * (2 * m1 + m2) * np.sin(θ1) - m2 * g * np.sin(θ1 - 2 * θ2) -
               2 * np.sin(θ1 - θ2) * m2 * (ω2**2 * l2 + ω1**2 * l1 * np.cos(θ1 - θ2))) / \
              (l1 * (2 * m1 + m2 - m2 * np.cos(2 * θ1 - 2 * θ2)))

        dω2 = (2 * np.sin(θ1 - θ2) * (ω1**2 * l1 * (m1 + m2) +
              g * (m1 + m2) * np.cos(θ1) + ω2**2 * l2 * m2 * np.cos(θ1 - θ2))) / \
              (l2 * (2 * m1 + m2 - m2 * np.cos(2 * θ1 - 2 * θ2)))

        return [ω1, dω1, ω2, dω2]

    def step(self, dt=0.01):
        sol = solve_ivp(self.dynamics, [self.t, self.t + dt], self.state, method='RK45', max_step=0.002)
        self.state = sol.y[:, -1]
        self.t += dt
        return self.state

# AI Model for Adaptive Control
class AdaptiveAI(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=5):  # Output size matches chaos params
        super(AdaptiveAI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

# AI-Driven Adaptation for ChaosCore
class AIEnhancedChaosCore:
    def __init__(self, chaos_engine):
        self.chaos_engine = chaos_engine
        self.ai_model = AdaptiveAI()
        self.optimizer = optim.Adam(self.ai_model.parameters(), lr=0.001)  # Faster convergence
        self.criterion = nn.MSELoss()
        self.prev_entropy = None

    def estimate_lyapunov_exponent(self, state_series):
        delta_x = np.abs(state_series[1:] - state_series[:-1])
        return np.mean(np.log(delta_x + 1e-8))  # Avoid log(0)

    def adapt(self, target_entropy):
        state_tensor = torch.tensor(self.chaos_engine.state, dtype=torch.float32).unsqueeze(0)
        predicted_adjustments = self.ai_model(state_tensor).squeeze(0)
        adjustments = torch.clamp(predicted_adjustments, -0.05, 0.05)

        if adjustments.shape[0] != self.chaos_engine.params.shape[0]:
            raise ValueError(f"Shape mismatch: predicted {adjustments.shape}, expected {self.chaos_engine.params.shape}")

        self.chaos_engine.params = np.clip(self.chaos_engine.params + adjustments.detach().numpy(), 0.1, 10.0)

        entropy = self.estimate_lyapunov_exponent(np.array(self.chaos_engine.state))
        entropy_diff = abs(entropy - target_entropy)
        dynamic_step = np.clip(entropy_diff * 0.1, 0.005, 0.1)

        self.chaos_engine.params = np.clip(self.chaos_engine.params + (adjustments.detach().numpy() * dynamic_step),
                                           0.1, 10.0)

        if self.prev_entropy is None:
            self.prev_entropy = entropy
        smoothed_entropy = 0.9 * self.prev_entropy + 0.1 * entropy
        self.prev_entropy = smoothed_entropy

        loss = self.criterion(
            torch.tensor([smoothed_entropy], dtype=torch.float32, requires_grad=True),
            torch.tensor([target_entropy], dtype=torch.float32, requires_grad=True),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), entropy

    def run(self, steps=100, target_entropy=0.99):
        for step in range(steps):
            self.chaos_engine.step()
            loss, entropy = self.adapt(target_entropy)
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss:.6f}, Entropy = {entropy:.6f}")
        return self.chaos_engine.state

# Initialize System
init_state = [np.pi/4, 0, np.pi/3, 0]
params = [1.0, 1.0, 1.0, 1.0, 9.81]
chaos_core = ChaosEngine(init_state, params)
adaptive_system = AIEnhancedChaosCore(chaos_core)

# Execute Adaptive ChaosCore
final_state = adaptive_system.run()
print("Final Chaos State:", final_state)
