"""
trainer.py contains the code for training and saving the model.
"""
import torch
import torch.nn as nn

from engine import train
from utils import save_model
from data_setup import load_data, load_vocab
from model import NeuralProbabilisticLanguageModel, Config


def main():
    # hyperparameters
    context_size = 5
    hidden_size = 100
    embed_size = 60
    direct = True
    epochs = 5
    vocab_file = "A Neural Probabilistic Language Model/data/vocab.pkl"

    # device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading data
    train_dataloader, dev_dataloader = load_data(context_size, vocab_file=vocab_file)
    vocab_size = len(load_vocab(vocab_file))
    print("Vocab size:", vocab_size)

    # defining model
    config = Config(vocab_size, embed_size, hidden_size, context_size, direct=direct)
    model = NeuralProbabilisticLanguageModel(config)
    model.to(device)

    # using AdamW optimizer with L2 regularization and a weight decay of 10e-4
    # learning rate is set to 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=10e-4)

    # using cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # training
    train(
        model=model,
        train_dataloader=train_dataloader,
        dev=dev_dataloader,
        optimizer=optimizer,
        loss_fn=criterion,
        epochs=epochs,
        device=device,
    )

    # defining a model name
    model_name = f"model_n{context_size}_h{hidden_size}_m{embed_size}"
    if direct:
        model_name += "_direct"
    save_file_name = model_name + ".pth"

    # saving model
    save_model(model, save_file_name)


if __name__ == "__main__":
    main()
