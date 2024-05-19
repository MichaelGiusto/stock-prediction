import torch
import csv
from lstm_spm_model import LSTM_SPM
import numpy as np
import random
import matplotlib.pyplot as plt

def prepare_sequences(stock_prices, seq_len=20):
    sequences = []
    labels = []

    for i in range(len(stock_prices) - seq_len):
        sequence = stock_prices[i:i + seq_len]
        sequences.append(sequence)
        labels.append(stock_prices[i + seq_len])

    return np.array(sequences), np.array(labels)


def load_aapl_data(file_path="stock_data/AAPL.csv"):
    input_file = csv.DictReader(open(file_path))
    data = []

    for row in input_file:
        data.append(row)
    
    price_data = []

    for row in data:
        price_data.append(float(row['Close']))
    
    return np.array(price_data)


def predict_next_days(model, sequence, days=10):
    model.eval()
    predictions = []
    current_seq = sequence

    for _ in range(days):
        with torch.no_grad():
            input_seq = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            next_price = model(input_seq).item()
            predictions.append(next_price)
            current_seq = np.append(current_seq[1:], next_price)

    return predictions


def main():
    model = LSTM_SPM()
    model.load_state_dict(torch.load("trained_spm.pth"))
    model.eval()

    price_data = load_aapl_data()
    sequences, labels = prepare_sequences(price_data)

    random_idx = random.randint(0, len(sequences) - 11)
    random_sequence = sequences[random_idx]

    predictions = predict_next_days(model, random_sequence, days=10)

    actual_prices = labels[random_idx:random_idx + 10]

    plt.figure(figsize=(10, 6))
    plt.plot(range(20), random_sequence, label="Actual Prices (Last 20 Days)")
    plt.plot(range(20, 30), predictions, label="Predicted Prices (Next 10 Days)", linestyle='--', marker='o')
    plt.plot(range(20, 30), actual_prices, label="Actual Prices (Next 10 Days)", linestyle='-', marker='x')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
