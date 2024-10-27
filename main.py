import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from utils import *

# Step 1: Load Images and Preprocess
print("Step 1: Load Images and Preprocess")
image_dir = '.\data\Images'  # Directory where images are located
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # List of jpg image files only

# Load labels from CSV
csv_path = '.\data\G1020.csv'
labels_df = pd.read_csv(csv_path)
labels_dict = dict(zip(labels_df['imageID'], labels_df['binaryLabels']))

images = []
labels = []

# Load and preprocess images
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    img = cv2.imread(image_path)
    if img is not None:
        img_resized = cv2.resize(img, (128, 128))  # Resize to 128x128
        img_normalized = img_resized / 255.0  # Normalize to range [0, 1]
        images.append(img_normalized)
        # Assign labels based on CSV file
        labels.append(labels_dict.get(image_file, 0))  # Default to 0 if not found

images = np.array(images).transpose(0, 3, 1, 2)  # Change shape to (N, C, H, W) for convolution
labels = np.array(labels)

# Step 2: Split Data into Training and Validation Sets
print("Step 2: Split Data into Training and Validation Sets")
def split_dataset(images, labels, train_ratio=0.8):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    train_size = int(len(images) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_images, val_images = images[train_indices], images[val_indices]
    train_labels, val_labels = labels[train_indices], labels[val_indices]

    return train_images, train_labels, val_images, val_labels

train_images, train_labels, val_images, val_labels = split_dataset(images, labels)

# Step 3: Define Helper Functions


# Step 4: Define Layers and Neural Network Classes
print("Step 4: Define Layers and Neural Network Classes")
class Convolution:
    def __init__(self, W, b, stride=1, pad=1):  # Adjusted pad to 1 for better alignment
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, _, H, W = x.shape
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        if col.shape[1] != col_W.shape[0]:
            raise ValueError(f"Dimension mismatch: col shape {col.shape}, col_W shape {col_W.shape}")
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout).transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):  # Adjusted stride to 2 for pooling layer
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.reshape(dout.shape[0], dout.shape[1], -1)  # Reshape to (N, C, H * W)
        pool_size = self.pool_h * self.pool_w

        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

class FullyConnected:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x.reshape(x.shape[0], -1)  # Flatten input for fully connected layer
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx.reshape(*self.x.shape)

# Step 5: Define Model Class
print("Step 5: Define Model Class")
class Model:
    def __init__(self):
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(16, 3, 3, 3) * 0.01,  # Convolution weights (FN, C, FH, FW)
            'b1': np.zeros(16),  # Convolution bias
            'W2': np.random.randn(64 * 64 * 16, 64) * 0.01,  # Fully connected weights (flattened output of pooling to hidden layer)
            'b2': np.zeros(64),
            'W3': np.random.randn(64, 1) * 0.01,  # Output weights (hidden layer to output)
            'b3': np.zeros(1)
        }
        self.layers = OrderedDict()
        self.layers['conv1'] = Convolution(self.params['W1'], self.params['b1'])
        self.layers['pool1'] = Pooling(2, 2, stride=2)
        self.layers['fc1'] = FullyConnected(self.params['W2'], self.params['b2'])
        self.layers['fc2'] = FullyConnected(self.params['W3'], self.params['b3'])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return sigmoid(x)

    def loss(self, x, t):
        y = self.predict(x)
        return binary_cross_entropy_error(y, t)

    def gradient(self, x, t):
        # Forward
        y = self.predict(x)

        # Backward (gradient calculation)
        dout = (y - t) / t.shape[0]
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Set gradients to params
        grads = {
            'W1': self.layers['conv1'].dW, 'b1': self.layers['conv1'].db,
            'W2': self.layers['fc1'].dW, 'b2': self.layers['fc1'].db,
            'W3': self.layers['fc2'].dW, 'b3': self.layers['fc2'].db
        }
        return grads

# Step 6: Training Loop
print("Step 6: Training Loop")
model = Model()
learning_rate = 0.003
epochs = 10
batch_size = 64
train_losses = []

for epoch in range(epochs):
    # Shuffle the training data
    permutation = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Mini-batch training
    epoch_loss = 0
    for i in range(0, train_images.shape[0], batch_size):
        x_batch = train_images[i:i + batch_size]
        t_batch = train_labels[i:i + batch_size].reshape(-1, 1)

        # Calculate gradients
        grads = model.gradient(x_batch, t_batch)

        # Update parameters
        for key in model.params:
            model.params[key] -= learning_rate * grads[key]

        # Loss calculation
        loss = model.loss(x_batch, t_batch)
        epoch_loss += loss

    # Record and print epoch loss
    train_losses.append(epoch_loss / (train_images.shape[0] // batch_size))
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]}')

print("Training complete.")

# Step 7: Plot Training Loss
print("Step 7: Plot Training Loss")
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

# Step 8: Evaluate on Validation Set
print("Step 8: Evaluate on Validation Set")
def evaluate(model, images, labels):
    correct_count = 0
    total_count = images.shape[0]
    batch_size = 64

    for i in range(0, total_count, batch_size):
        x_batch = images[i:i + batch_size]
        t_batch = labels[i:i + batch_size].reshape(-1, 1)

        # Forward pass
        out = model.layers['conv1'].forward(x_batch)
        out = model.layers['pool1'].forward(out)
        out = out.reshape(out.shape[0], -1)
        out = model.layers['fc1'].forward(out)
        out = model.layers['fc2'].forward(out)
        preds = sigmoid(out)

        # Binarize predictions and compare to true labels
        preds_binary = (preds > 0.5).astype(int)
        correct_count += np.sum(preds_binary == t_batch)

    accuracy = correct_count / total_count
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model on the validation set
evaluate(model, val_images, val_labels)
