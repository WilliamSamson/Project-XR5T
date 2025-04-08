import torch
import torch.nn as nn

# Each shell is a neural trait mesh — learnable logic/emotion/ethics
class ShellCore(nn.Module):
    def __init__(self, trait_dim=12):
        super().__init__()
        self.logic_net = nn.Linear(trait_dim, trait_dim)
        self.emotion_net = nn.GRU(trait_dim, trait_dim)
        self.ethics_net = nn.Linear(trait_dim, 1)
        self.entropy = torch.randn(1)  # Entropy vector for decay/mutation
        self.shadow_influence = 0.0

    def forward(self, traits, memory_bias):
        logic = self.logic_net(traits)
        emotion_output, _ = self.emotion_net(traits.unsqueeze(0))
        ethics = torch.sigmoid(self.ethics_net(logic))
        return logic + emotion_output.squeeze(0) + ethics + memory_bias

# MetaShell learns shell usage patterns, spawns new agents
class ExoGhost(nn.Module):
    def __init__(self, shell_registry):
        super().__init__()
        self.shell_registry = shell_registry
        self.trait_embedding = nn.Embedding(len(shell_registry), 12)
        self.meta_controller = nn.GRU(12, 12)
        self.output_layer = nn.Linear(12, len(shell_registry))

    def forward(self, usage_log):
        embedded = self.trait_embedding(usage_log)
        output, _ = self.meta_controller(embedded)
        priorities = torch.softmax(self.output_layer(output[-1]), dim=0)
        return priorities  # Determines which shell should activate

    def forge_shell(self, input_vector, entropy):
        mutated = input_vector + torch.randn_like(input_vector) * entropy
        new_shell = ShellCore()
        new_shell.entropy = entropy
        return new_shell