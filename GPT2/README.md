# GPT2

This folder contains a trained GPT2 model with BPE tokenization trained on Tiny-Stories.

## Dataset
The [Tiny-Stories dataset](https://arxiv.org/abs/2305.07759) is a collection of short stories used to train the GPT2 model.

## Tokenization
`tokenizer.py` contains my implementation of BPE GPT4o tokenizer from scratch.
A pretrained tokenizer on the TinyStories dataset is available on the Huggingface model hub [here](https://huggingface.co/aspisov/gpt2-tinystories-tokenizer).

## Usage
To use the pretrained tokenizer, follow these steps:
1. Install the required libraries:
	```bash
	pip install transformers
	```
2. Load the model and tokenizer:
	```python
	from transformers import AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained('aspisov/gpt2-tinystories-tokenizer')
	```

## Model

# References
- [YSDA Lena Voita lecture](https://github.com/yandexdataschool/nlp_course/tree/2024/week04_seq2seq)
- [Attention is all you need paper](https://arxiv.org/abs/1706.03762)
- [GPT2 paper](https://openai.com/index/better-language-models/)
- [Andrej Karpathy tokenizer video](https://youtu.be/zduSFxRajkE?si=xwqQ4F-VT0zHoRvc)
