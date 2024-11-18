import torch
from argparse import ArgumentParser
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from src.components.ffn import FF_Net
from src.components.cnn import Conv_Net
from src.components.data_loader import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = ArgumentParser()
args.add_argument('--batch_size', type=int, default=128)
args.add_argument('--epochs', type=int, default=10)
args.add_argument('--model', type=str, default='ffn')
args.add_argument('--checkpoint_dir', type=str, default='checkpoints')
args.parse_args()

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            if args.model == 'ffn':
                inputs = inputs.view(inputs.shape[0], -1)
                
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy

def train():
    # Initialize wandb
    wandb.init(project="advanced-training", config=vars(args))
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train_loader, test_loader = load_data(args.batch_size)

    model = FF_Net() if args.model == 'ffn' else Conv_Net()
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if args.model == 'ffn':
                inputs = inputs.view(inputs.shape[0], -1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()

            if i % 100 == 0:
                # Log training metrics
                wandb.log({
                    "train_batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "batch": i
                })
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        
        # Log test metrics
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "epoch": epoch
        })
        
        # Adjust learning rate based on test loss
        scheduler.step(test_loss)
        
        # Save checkpoint if test loss improved
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved new best model with test loss: {best_loss:.4f}")

    wandb.finish()
