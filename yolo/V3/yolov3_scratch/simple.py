import torch
import torch.nn as nn

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # Fully connected layer

    def forward(self, x):
        x = self.fc(x)
        return x

# Instantiate the model
model = SimpleModel()

# Check if GPU is available
if torch.cuda.is_available():
    # Move the model to GPU
    model.cuda()
    print("Model moved to GPU.")
else:
    print("GPU not available. Using CPU.")

# Print the model architecture
print(model)
