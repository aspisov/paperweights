"""
model.py contains the code for the NPLM model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int
    embed_size: int
    hidden_size: int
    context_size: int
    direct: bool

class NeuralProbabilisticLanguageModel(nn.Module):
    def __init__(self, config: Config):
        """
        Initializes a NeuralProbabilisticLanguageModel object.

        Args:
            config (Config): The configuration object containing the necessary parameters for the model.

        Initializes the following attributes:
            - embed_size (int): The size of the embeddings.
            - context_size (int): The size of the context.
            - direct (bool): Whether to use direct connections.
            - C (nn.Embedding): The embedding layer.
            - H (nn.Linear): The hidden layer.
            - tanh (nn.Tanh): The tanh activation function.
            - U (nn.Linear): The output layer.
            - W (nn.Linear, optional): The linear layer for direct connections.

        Calls the following methods:
            - _init_weights: Initializes all weights in the model.

        Prints the number of parameters in the model.
        """
        super().__init__()
        self.embed_size = config.embed_size
        self.context_size = config.context_size
        self.direct = config.direct

        self.C = nn.Embedding(config.vocab_size, config.embed_size)
        self.H = nn.Linear(config.context_size * config.embed_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.U = nn.Linear(config.hidden_size, config.vocab_size)
        
        if config.direct:
            self.W = nn.Linear(config.context_size * config.embed_size, config.vocab_size, bias=False)
        
        # Initialize all weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def forward(self, x):
        """
        Performs the forward pass of the neural network model.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The output logits of the model.
        """
        x = self.C(x.to(torch.long))  # Convert input tokens to embeddings
        x = x.view(-1, self.context_size * self.embed_size)  # Flatten for linear layers
        
        hidden = self.tanh(self.H(x))  # Hidden layer processing
        
        if self.direct:
            logits = self.W(x) + self.U(hidden)  # Compute logits from output of different layers
        else:
            logits = self.U(hidden)
        return logits

    def get_num_params(self):
        # Return the total number of parameters in the model
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        # Custom weight initialization
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        
