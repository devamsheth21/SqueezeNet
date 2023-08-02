import torch

def evaluate_model(model, test_loader):
    # Load the trained model checkpoint
    model.load_state_dict(torch.load('model_checkpoint.pth'))

    # Set the model in evaluation mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
