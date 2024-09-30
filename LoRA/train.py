import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from datasets import load_dataset

MODEL_NAME = "gpt2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)

for param in model.parameters():
    param.requires_grad = False


class LoRA(nn.Module):
    """Low-Rank Adaptation (LoRA) layer."""

    def __init__(self, module, rank, dropout=0.2):
        """Initializes the LoRA layer.

        Args:
            module: The original module to be adapted.
            rank: The rank for the low-rank adaptation.
            dropout: Dropout rate.
        """
        super(LoRA, self).__init__()
        self.module = module
        in_features, out_features = module.weight.shape
        self.adapter_A = nn.Parameter(
            torch.empty(in_features, rank, device=module.weight.device)
        )
        self.adapter_B = nn.Parameter(
            torch.empty(rank, out_features, device=module.weight.device)
        )
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes the weights of the adapters."""
        nn.init.normal_(self.adapter_A, mean=0, std=0.02)
        nn.init.zeros_(self.adapter_B)

    def forward(self, x):
        """Forward pass through the LoRA layer."""
        return self.module(x) + self.dropout(
            x @ self.adapter_A @ self.adapter_B
        )


LORA_RANK = 8

# Insert LoRA layers into the model
for i in range(len(model.transformer.h)):
    model.transformer.h[i].mlp.c_fc = LoRA(
        model.transformer.h[i].mlp.c_fc, rank=LORA_RANK
    ).to(DEVICE)
    model.transformer.h[i].mlp.c_proj = LoRA(
        model.transformer.h[i].mlp.c_proj, rank=LORA_RANK
    ).to(DEVICE)
    model.transformer.h[i].attn.c_attn = LoRA(
        model.transformer.h[i].attn.c_attn, rank=LORA_RANK
    ).to(DEVICE)


# ---------------------- Training the Model ---------------------- #


dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)


def collate_fn(batch):
    """Collate function for the DataLoader.

    Args:
        batch: List of samples.

    Returns:
        Tokenized and padded batch.
    """
    texts = [item["text"] for item in batch]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)


train_dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)


def train_model(model, dataloader, epochs=3, lr=1e-4):
    """Trains the model.

    Args:
        model: The model to be trained.
        dataloader: The DataLoader for the training data.
        epochs: Number of training epochs.
        lr: Learning rate.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, Training Loss: {loss.item()}")

    model.eval()


def generate_text(prompt, max_length=50):
    """Generates text based on a prompt.

    Args:
        prompt: The input text prompt.
        max_length: Maximum length of the generated text.

    Returns:
        Generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


train_model(model, train_dataloader)

prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(f"Generated text: {generated_text}")
