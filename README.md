# sign-language-
A machine learning-based system to recognize and translate sign language gestures into text or speech. Uses Python, TensorFlow, Keras, and OpenCV for training models and gesture recognition.
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d grassknoted/asl-alphabet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import pydot
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

uniq_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space', 'nothing'
]
def load_training_dataset(directory, uniq_labels):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        label_directory = os.path.join(directory, label)
        if not os.path.exists(label_directory):
            print(f"Warning: {label_directory} does not exist. Skipping.")
            continue

        for file in os.listdir(label_directory):
            filepath = os.path.join(label_directory, file)
            img = cv2.imread(filepath)
            if img is None:
                print(f"Warning: Could not read {filepath}. Skipping.")
                continue

            img = cv2.resize(img, (50, 50))
            images.append(img)
            labels.append(idx)

    images = np.asarray(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    labels = np.asarray(labels)
    return images, labels
    def load_test_dataset(directory, uniq_labels):
    images = []
    labels = []

    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Warning: Could not read {filepath}. Skipping.")
            continue

        img = cv2.resize(img, (50, 50))
        images.append(img)

        # Extract label from the filename (remove '_test' and convert to uppercase)
        label_name = os.path.splitext(file)[0].replace('_test', '').upper()

        # Adjust for 'nothing' and 'space' labels
        if label_name == 'NOTHING':
            label_name = 'nothing'
        elif label_name == 'SPACE':
            label_name = 'space'

        if label_name in uniq_labels:
            label_index = uniq_labels.index(label_name)
            labels.append(label_index)
        else:
            print(f"Warning: Label '{label_name}' not found in unique labels. Skipping.")

    images = np.asarray(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    labels = np.asarray(labels)
    return images, labels
    

train_directory = '/content/asl_alphabet_train/asl_alphabet_train/'  # Correct path
test_directory = '/content/asl_alphabet_test/asl_alphabet_test/'    # Correct path

train_images, train_labels = load_training_dataset(train_directory, uniq_labels)
test_images, test_labels = load_test_dataset(test_directory, uniq_labels)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagen.fit(train_images)
# Define your custom model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(uniq_labels), activation='softmax') 
    # Output layer
])

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Convert labels to one-hot encoded vectors
num_classes = len(uniq_labels)  # Set to the number of unique labels

Y_train = to_categorical(train_labels, num_classes=num_classes)  # Training labels
Y_test = to_categorical(test_labels, num_classes=num_classes)    # Test labels

# Print shapes to verify
print("Shapes after one-hot encoding:")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Fit the model
history = model.fit(
    train_images,
    Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(test_images, Y_test)
)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, Y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

print(f"Unique classes in test predictions: {len(np.unique(test_predictions_classes))}")

valid_labels = uniq_labels[:28]

print(classification_report(test_labels, test_predictions_classes, target_names=valid_labels))

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Compute confusion matrix
conf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), test_predictions_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
#saving the model for further use.

model.save('hand_sign_model.h5')
print("Model saved successfully.")

def display_all_images(images, labels, uniq_labels, batch_size=10, dataset_type="Test"):
    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch in range(num_batches):
        plt.figure(figsize=(15, 15))
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_images)
        for i in range(start_idx, end_idx):
            plt.subplot(1, batch_size, (i - start_idx) + 1)
            plt.imshow(images[i])
            plt.title(f'{dataset_type} - {uniq_labels[labels[i]]}')
            plt.axis('off')
        plt.show()

display_all_images(test_images, test_labels, uniq_labels, batch_size=10, dataset_type="Test")

# Class distribution in the training set
plt.figure(figsize=(10, 5))
plt.bar(uniq_labels, np.bincount(train_labels), color='b')
plt.title('Training Set Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Class distribution in the test set
plt.figure(figsize=(10, 5))
plt.bar(uniq_labels, np.bincount(test_labels), color='g')
plt.title('Test Set Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Visualize correct and incorrect predictions
def visualize_predictions(images, true_labels, predicted_labels, uniq_labels, num_images=5):
    correct = np.where(true_labels == predicted_labels)[0]
    incorrect = np.where(true_labels != predicted_labels)[0]

    # Plot correct predictions
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(correct[:num_images]):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[idx])
        plt.title(f"True: {uniq_labels[true_labels[idx]]}\nPred: {uniq_labels[predicted_labels[idx]]}")
        plt.axis('off')
    plt.suptitle('Correct Predictions', size=16)
    plt.show()

    # Plot incorrect predictions
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(incorrect[:num_images]):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[idx])
        plt.title(f"True: {uniq_labels[true_labels[idx]]}\nPred: {uniq_labels[predicted_labels[idx]]}")
        plt.axis('off')
    plt.suptitle('Incorrect Predictions', size=16)
    plt.show()

# Call the function to visualize correct and incorrect predictions
visualize_predictions(test_images, np.argmax(Y_test, axis=1), test_predictions_classes, uniq_labels)

def visualize_filters(layer, model):
    filters, biases = model.layers[layer].get_weights()
    n_filters = filters.shape[-1]
    plt.figure(figsize=(10, 10))
# Visualizing the first 16 filters
    for i in range(min(n_filters, 16)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(filters[:, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.show()
visualize_filters(0, model)

def display_random_images(images, labels, uniq_labels, num_images=10, dataset_type="Train"):
    indices = np.random.randint(0, len(images), num_images)
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[idx])
        plt.title(f'{dataset_type} - {uniq_labels[labels[idx]]}')
        plt.axis('off')
    plt.show()
display_random_images(train_images, train_labels, uniq_labels, num_images=10, dataset_type="Train")
display_random_images(test_images, test_labels, uniq_labels, num_images=10, dataset_type="Test")
