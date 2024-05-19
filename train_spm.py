import torch
import csv
from lstm_spm_model import LSTM_SPM
from torch.utils.data import DataLoader
import numpy as np
from spm_dataset import SPMDataset
import torch.nn as nn
import torch.optim as optim
import os


def prepare_sequences(stock_prices):
    sequences = []
    labels = []

    for i in range(len(stock_prices) - 20):
        sequence = stock_prices[i:i + 20]
        sequences.append(sequence)
        labels.append(stock_prices[i + 20])

    return np.array(sequences), np.array(labels)


def load_data():
    total_sequences = []
    total_labels = []
    for paths, dirs, files in os.walk("stock_data/"):
        for file in files:
            input_file = csv.DictReader(open(os.path.join(paths, file)))
            data = []
            
            for row in input_file:
                data.append(row)

            price_data = []

            for row in data:
                price_data.append(float(row['Close']))
            
            sequences, labels = prepare_sequences(price_data)
            
            total_sequences.extend(sequences)
            total_labels.extend(labels)

    return np.array(total_sequences), np.array(total_labels)


def train(model, dataloader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')


def main():
    sequences, labels = load_data()

    dataset_train = SPMDataset(sequences, labels)
    dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)

    model = LSTM_SPM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion, optimizer)

    torch.save(model.state_dict(), "trained_spm.pth")


if __name__ == '__main__':
    main()