import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken


@dataclass
class Config:
    """Configuration for the GPT model."""

    context_window: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 8
    embedding_dim: int = 768


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(self, config: Config):
        """Initializes the MLP module.

        Args:
            config: A Config instance containing model hyperparameters.
        """
        super(MLP, self).__init__()
        self.config = config

        self.c_fc = nn.Linear(config.embedding_dim, config.embedding_dim * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.embedding_dim * 4, config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):
    """Self-Attention module."""

    def __init__(self, config: Config):
        """Initializes the Self-Attention module.

        Args:
            config: A Config instance containing model hyperparameters.
        """
        super(SelfAttention, self).__init__()
        self.config = config

        self.c_attn = nn.Linear(config.embedding_dim, config.embedding_dim * 3)
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.context_window, config.context_window)
            ).view(1, 1, config.context_window, config.context_window),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Self-Attention module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        B, T, C = x.size()

        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=-1)
        k = k.view(
            B, T, self.config.num_heads, C // self.config.num_heads
        ).transpose(
            1, 2
        )  # (B, num_heads, T, C // num_heads)
        v = v.view(
            B, T, self.config.num_heads, C // self.config.num_heads
        ).transpose(
            1, 2
        )  # (B, num_heads, T, C // num_heads)
        q = q.view(
            B, T, self.config.num_heads, C // self.config.num_heads
        ).transpose(
            1, 2
        )  # (B, num_heads, T, C // num_heads)

        att = q @ k.transpose(-2, -1) / (C // self.config.num_heads) ** 0.5
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, num_heads, T, C // num_heads)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)

        return y


class Block(nn.Module):
    """A single transformer block."""

    def __init__(self, config: Config):
        """Initializes the transformer block.

        Args:
            config: A Config instance containing model hyperparameters.
        """
        super(Block, self).__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the transformer block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT model."""

    def __init__(self, config: Config):
        """Initializes the GPT model with the given configuration.

        Args:
            config: A Config instance containing model hyperparameters.
        """
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.embedding_dim),
                "wpe": nn.Embedding(
                    config.context_window, config.embedding_dim
                ),
                "h": nn.ModuleList(
                    [Block(config) for _ in range(config.num_layers)]
                ),
                "ln_f": nn.LayerNorm(config.embedding_dim),
            }
        )

        self.lm_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GPT model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        B, T = x.size()

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)

        token_emb = self.transformer.wte(x)  # (B, T, embedding_dim)
        pos_emb = self.transformer.wpe(pos)  # (T, embedding_dim)
        x = token_emb + pos_emb  # (B, T, embedding_dim)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Loads a pretrained GPT model.

        Args:
            model_type: The type of GPT model to load.
            override_args: Optional dictionary of arguments to override.

        Returns:
            A GPT model instance.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2": dict(num_layers=12, num_heads=12, embedding_dim=768),
            "gpt2-medium": dict(
                num_layers=24, num_heads=16, embedding_dim=1024
            ),
            "gpt2-large": dict(num_layers=36, num_heads=20, embedding_dim=1280),
            "gpt2-xl": dict(num_layers=48, num_heads=25, embedding_dim=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["context_window"] = 1024
        config = Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [
            k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")
        ]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# --------------------------------------------
def generate_text(
    model, prompt, num_return_sequences=5, max_length=30, top_k=50
):
    """Generates text using the GPT model.

    Args:
        model: The GPT model.
        prompt: The initial text prompt.
        num_return_sequences: Number of sequences to return.
        max_length: Maximum length of the generated sequences.
        top_k: Number of top tokens to sample from.

    Returns:
        A list of generated text sequences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    while x.size(1) < max_length:
        with torch.inference_mode():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # top-k sampling
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    generated_sequences = []
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        generated_sequences.append(decoded)

    return generated_sequences
