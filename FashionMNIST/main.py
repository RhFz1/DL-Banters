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

    correct_ffn = 0
    total_ffn = 0

    correct_cnn = 0
    total_cnn = 0

    with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
        size = len(testloader.dataset)
        num_batches = len(testloader)

        print(f"Size of the DS: {size}, Number of batches: {num_batches}")

        test_loss, correct = 0.0, 0

        for i, data in enumerate(testloader, 0):

            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            
            outputs_cnn = conv_net(inputs)
            # Flatten inputs for ffn
            inputs = inputs.view(inputs.shape[0], -1)
            outputs_ffn = feedforward_net(inputs)
            
            probs_ffn = F.softmax(outputs_ffn, dim=1)
            probs_cnn = F.softmax(outputs_cnn, dim=1)

            correct_ffn += (probs_ffn.argmax(1) == labels).type(torch.float).sum().item()
            total_ffn += labels.size(0)
            correct_cnn += (probs_cnn.argmax(1) == labels).type(torch.float).sum().item()
            total_cnn += labels.size(0)

    print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
    print('Accuracy for convolutional network: ', correct_cnn/total_cnn)

