import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from src.components.cnn import *
from src.components.ffn import *

'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''

'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform_train = transforms.Compose([
                # Random horizontal flip
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomErasing(p=0.1),
                # Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

batch_size = 128
device= 'cuda'

'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size // 4, shuffle=False)


'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''


feedforward_net = FF_Net()
conv_net = Conv_Net()

feedforward_net.to(device)
conv_net.to(device)

'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = F.cross_entropy

optimizer_ffn = optim.AdamW(feedforward_net.parameters(), lr=3e-4, weight_decay=2e-4,betas=(0.9, 0.999))
optimizer_cnn = optim.AdamW(conv_net.parameters(), lr=3e-4, weight_decay=6e-4, betas=(0.9, 0.999))



'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''


num_epochs_ffn = 30

lossf = []


for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0
    itr = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

        # Flatten inputs for ffn

        inputs = inputs.view(inputs.shape[0], -1)

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs,labels) 
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()
        itr += 1
        lossf.append(loss.item())
        if i % 2 == 1:
            print(f"EPOCH: {epoch}, BATCH: {i}, LOSS: {loss.item():.4f}")
    avg_loss = running_loss_ffn/itr
    
    print(f"Epoch {epoch}: {running_loss_ffn/itr:.4f}")
print('Finished Training')
# Move the model to the CPU before saving
torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)

fig = plt.figure()
plt.plot(torch.arange(len(lossf)), lossf)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration for Feedforward Network')
plt.savefig('ffn_loss.png', bbox_inches='tight')  # bbox_inches='tight' removes extra whitespace
plt.close(fig)  # Explicitly close the figure to free memory

num_epochs_cnn = 20
lossc = []
for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0
    itr = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()
        lossc.append(loss.item())
        itr += 1
    
    avg_loss = running_loss_cnn/itr
    print(f"Epoch:{epoch}, Training loss: {running_loss_cnn/itr:.4f}")

print('Finished Training')
torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)

fig = plt.figure()
plt.plot(torch.arange(len(lossc)), lossc)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration for Convolutional Network')
plt.savefig('cnn_loss.png', bbox_inches='tight')  # bbox_inches='tight' removes extra whitespace
plt.close(fig)  # Explicitly close the figure to free memory




'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

feedforward_net.load_state_dict(torch.load('ffn.pth'))
conv_net.load_state_dict(torch.load('cnn.pth'))

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


# Define the class labels for FashionMNIST
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_predictions(model, is_cnn=False, name="model"):
    model.eval()
    found_correct = False
    found_incorrect = False
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            if not is_cnn:
                images_flat = images.view(images.shape[0], -1)
                outputs = model(images_flat)
            else:
                outputs = model(images)
                
            probs = F.softmax(outputs, dim=1)
            predictions = probs.argmax(1)
            
            # Find one correct and one incorrect prediction
            for i in range(len(predictions)):
                pred = predictions[i]
                true_label = labels[i]
                
                if pred == true_label and not found_correct:
                    # Plot correct prediction
                    img = images[i].cpu().squeeze()
                    axes[0].imshow(img, cmap='gray')
                    axes[0].set_title(f'Correct Prediction\nPredicted: {class_labels[pred]}\nTrue: {class_labels[true_label]}')
                    axes[0].axis('off')
                    found_correct = True
                    
                elif pred != true_label and not found_incorrect:
                    # Plot incorrect prediction
                    img = images[i].cpu().squeeze()
                    axes[1].imshow(img, cmap='gray')
                    axes[1].set_title(f'Incorrect Prediction\nPredicted: {class_labels[pred]}\nTrue: {class_labels[true_label]}')
                    axes[1].axis('off')
                    found_incorrect = True
                    
            if found_correct and found_incorrect:
                break
    
    plt.tight_layout()
    plt.savefig(f'{name}_predictions.png', bbox_inches='tight', dpi=300)
    plt.close()

# Generate prediction plots for both models
plot_predictions(feedforward_net, is_cnn=False, name="ffn")
plot_predictions(conv_net, is_cnn=True, name="cnn")