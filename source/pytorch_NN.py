import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

def f(x, sd=0):
    return x**4+3*x**3 + 5*x**2 + 2*x - 1 + random.gauss(0, sd)

xrange = (-300, 300)
samples = 1000
d = (xrange[1] - xrange[0]) / samples
X = [xrange[0] + i*d for i in range(samples)]
Y = [f(x) for x in X]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model with BatchNorm, Dropout, and Xavier initialization
class CustomMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            linear = nn.Linear(in_features, out_features)
            
            # Xavier (Glorot) initialization
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            
            layers.append(linear)
            
            if i < len(layer_sizes) - 2:
                # Activation
                layers.append(nn.ReLU(inplace=True))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Define layer sizes: [input_dim, ..., output_dim]
layer_sizes = [1, 30, 300, 3000, 1]

# Create the model
model = CustomMLP(layer_sizes).to(device)

lr = 0.01
optimizer = Adam(model.parameters(), lr=lr, weight_decay=lr/100)

# Mean Squared Error loss
criterion = nn.MSELoss()

# Convert data to tensors and move to GPU if available
X_t = torch.tensor(X).unsqueeze(1).float().to(device)
Y_t = torch.tensor(Y).unsqueeze(1).float().to(device)

# Training loop
epochs = 3001
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    Ypred = model(X_t)
    
    # Compute data loss
    data_loss = criterion(Ypred, Y_t)
    
    # Total loss
    total_loss = data_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Print loss periodically
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Data Loss: {data_loss.item()}")

# Plotting the results
model.eval()
with torch.no_grad():
    Ypred_cpu = model(X_t).cpu().numpy()

X_cpu = X_t.cpu().numpy()
Y_cpu = Y_t.cpu().numpy()

plt.scatter(X_cpu, Y_cpu, label='True')
plt.scatter(X_cpu, Ypred_cpu, label='Predicted')
plt.title('MLP Regression')
plt.legend()
plt.show()