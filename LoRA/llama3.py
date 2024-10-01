# ---------------------------- Importing Libraries --------------------------- #

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

# ---------------------------- Logging into HF ---------------------------- #

hf_token = "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"
login(token=hf_token)

# ---------------------------- Loading the model ---------------------------- #

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "John gave beer to Marry, she returned it back to"

input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
output_ids = model.generate(input_ids, max_length=50)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)

# ---------------------------- LoRA layer ---------------------------- #


class LoRALayer(nn.Module):
    def __init__(self, r, alpha, dropout=0):
        super(LoRALayer, self).__init__()
        self.r = r
        self.alpha = alpha
        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
