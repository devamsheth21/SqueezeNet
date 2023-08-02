import torch
import torch.optim as optim

def train_model(model, train_loader):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set the model in training mode
    model.train()

    num_epochs = 10  # Define the number of epochs

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 1):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:  # Adjust the logging frequency as desired
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {running_loss / i:.4f}')

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Save the trained model checkpoint
    torch.save(model.state_dict(), 'model_checkpoint.pth')

    print('Training complete.')
