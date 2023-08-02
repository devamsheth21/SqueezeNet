import torch
from data_loader import load_dataset
from train import train_model
from evaluate import evaluate_model
from squeezenet import SqueezeNet

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load dataset
    train_loader, test_loader = load_dataset()

    # Initialize SqueezeNet model
    model = SqueezeNet()

    # Train the model
    train_model(model, train_loader)

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()
