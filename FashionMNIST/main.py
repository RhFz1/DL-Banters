import torch
from src.components.ffn import FF_Net
from src.components.cnn import Conv_Net
from src.components.data import load_data


def main():

    # Load data
    train_loader, test_loader = load_data()

    # Initialize model
    cnn_model = Conv_Net()
    ffn_model = FF_Net()

    # Loading the trained model
    cnn_model.load_state_dict(torch.load('artifacts/cnn.pth'))
    ffn_model.load_state_dict(torch.load('artifacts/ffn.pth'))

