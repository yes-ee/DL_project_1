import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from model import Model

# Step 1: Load Images and Preprocess
print("Step 1: Load Images and Preprocess")
image_dir = '.\data\Images'  # Directory where images are located
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # List of jpg image files only

# Load labels from CSV
csv_path = '.\data\G1020.csv'
labels_df = pd.read_csv(csv_path)
labels_dict = dict(zip(labels_df['imageID'], labels_df['binaryLabels']))

# count label 0
count_0 = 0
total_0 = int(sum(labels_df['binaryLabels'] == 0) * 0.7)

images = []
labels = []

print(total_0)

# Load and preprocess images
for image_file in image_files:
    label = labels_dict.get(image_file, 0)

    # If label is 0, check the counter to load 70%
    if label == 0:
        if count_0 >= total_0:
            continue  # Skip extra 0-label images
        count_0 += 1

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

print(images.shape)


def balance_dataset(images, labels, target_ratio=0.5, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)

    # Separate indices for each class
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]

    # Calculate target number of samples based on the smaller class size and target ratio
    min_class_size = min(len(class_0_indices), len(class_1_indices))
    target_class_size = min(int(min_class_size / target_ratio), len(class_0_indices), len(class_1_indices))

    # Randomly sample from each class based on the target class size with a fixed seed
    balanced_class_0_indices = np.random.choice(class_0_indices, target_class_size, replace=False)
    balanced_class_1_indices = np.random.choice(class_1_indices, target_class_size, replace=False)

    # Concatenate indices from both classes and shuffle
    balanced_indices = np.concatenate([balanced_class_0_indices, balanced_class_1_indices])
    np.random.shuffle(balanced_indices)

    # Create a balanced dataset of images and labels
    balanced_images = images[balanced_indices]
    balanced_labels = labels[balanced_indices]

    return balanced_images, balanced_labels


# Step 2: Split Data into Training and Validation Sets
def split_dataset(images, labels, train_ratio=0.8):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    train_size = int(len(images) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_images, val_images = images[train_indices], images[val_indices]
    train_labels, val_labels = labels[train_indices], labels[val_indices]

    return train_images, train_labels, val_images, val_labels


# Apply dataset balancing
balanced_images, balanced_labels = balance_dataset(images, labels, target_ratio=0.5)

# Split the balanced dataset into training and validation sets
train_images, train_labels, val_images, val_labels = split_dataset(balanced_images, balanced_labels, train_ratio=0.8)

# Step 3: Define Helper Functions


# Step 4: Define Layers and Neural Network Classes
print("Step 4: Define Layers and Neural Network Classes")


# Step 5: Define Model Class
print("Step 5: Define Model Class")

# Instantiate the model

# Step 6: Define Evaluate function


# Step 7: Training Loop and Evaluate
model = Model()

learning_rate = 0.005
epochs = 10
batch_size = 64
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    # Shuffle the training data
    permutation = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Mini-batch training
    epoch_loss = 0
    correct_count = 0
    for i in range(0, train_images.shape[0], batch_size):
        x_batch = train_images[i:i + batch_size]
        t_batch = train_labels[i:i + batch_size].reshape(-1, 1)

        # Calculate gradients
        grads = model.gradient(x_batch, t_batch)

        # Update weights
        for key in model.params:
            model.params[key] -= learning_rate * grads[key]

        # Calculate loss
        loss = model.loss(x_batch, t_batch)
        epoch_loss += loss

        # Calculate accuracy
        out = model.predict(x_batch)
        preds_binary = (out > 0.5).astype(int)
        correct_count += np.sum(preds_binary == t_batch)

    train_accuracy = correct_count / train_images.shape[0]
    train_avg_loss = epoch_loss / (train_images.shape[0] // batch_size)
    train_losses.append(train_avg_loss)
    train_accuracies.append(train_accuracy)

    # Evaluate on validation set
    val_accuracy, val_avg_loss = evaluate(model, val_images, val_labels)
    val_losses.append(val_avg_loss)
    val_accuracies.append(val_accuracy)

    # Print epoch summary
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_avg_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, ' +
          f'Validation Loss: {val_avg_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')


# Step 8: Plot Training and Validation Accuracy
plt.plot(range(1, epochs + 1), train_accuracies, marker='o', label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, marker='s', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()

# Step 9: Final Evaluation on Validation Set
evaluate(model, val_images, val_labels)
