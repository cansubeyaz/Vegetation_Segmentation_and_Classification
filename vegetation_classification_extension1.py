import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from keras import initializers
import matplotlib.pyplot as plt
import argparse
import time
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Vegetation Multiclass Classification using CNN')
    parser.add_argument('image_path', type=str, nargs='?', default=None, help='Image file for prediction')
    parser.add_argument('--data_dir', type=str, help='Directory', default='PART_II_DATA')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer', default=0.00001)
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=125) ## also 400 epochs tested
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=40)
    parser.add_argument('--optimizer', type=str, help='Optimizer to use for training', choices=['adam', 'sgd', 'rmsprop'], default='adam')
    parser.add_argument('--loss', type=str, help='Loss function', default='sparse_categorical_crossentropy')
    parser.add_argument('--model_path', type=str, help='Save or load the model', default='vegetation_cnn_extension_model.pkl')
    return parser.parse_args()

def load_images_from_folder(folder, target_shape=(81, 81, 3)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img) / 255.
            if img_array.shape == target_shape:
                images.append(img_array)
            else:
                print(f"Image {filename} due to shape mismatch: {img_array.shape}")
    return np.array(images)

def load_data(data_dir): ## Load data from directories
    class_labels = {'crop': 1, 'grass': 0, 'other': 2} ## Multiclass classification
    train_data, train_labels = [], []
    valid_data, valid_labels = [], []
    eval_data, eval_labels = [], []

    for cls, label in class_labels.items():
        train_folder = os.path.join(data_dir, cls, 'train')
        valid_folder = os.path.join(data_dir, cls, 'valid')
        eval_folder = os.path.join(data_dir, cls, 'eval')

        train_images = load_images_from_folder(train_folder) ## Load the training images
        valid_images = load_images_from_folder(valid_folder) ## Load the validation images
        eval_images = load_images_from_folder(eval_folder) ## Load the evaluation images

        train_data.extend(train_images)
        train_labels.extend([label] * len(train_images)) ## training label data
        valid_data.extend(valid_images)
        valid_labels.extend([label] * len(valid_images)) ## validation label data
        eval_data.extend(eval_images)
        eval_labels.extend([label] * len(eval_images))  ## evaluation label data

        print(f'Class: {cls}')
        print(f'  Training images: {len(train_images)}')
        print(f'  Validation images: {len(valid_images)}')
        print(f'  Evaluation images: {len(eval_images)}\n')

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    valid_data = np.array(valid_data)
    valid_labels = np.array(valid_labels)
    eval_data = np.array(eval_data)
    eval_labels = np.array(eval_labels)
    print(f'FULL DATA:')
    print(f'  Training images: {len(train_data)}, {len(train_labels)}')
    print(f'  Validation images: {len(valid_data)}, {len(valid_labels)}')
    print(f'  Evaluation images: {len(eval_data)}, {len(eval_labels)}\n')

    return train_data, train_labels, valid_data, valid_labels, eval_data, eval_labels

def normalize_data(train_data, valid_data, eval_data): ## Normalization: Mean and std calculated from training data
    mean, std = train_data.mean(), train_data.std()
    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    eval_data = (eval_data - mean) / std
    return train_data, valid_data, eval_data, mean, std

# Function to build a CNN model
def build_cnn_model(learning_rate, optimizer_name, loss_name): ## CNN model with random initialization
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(81, 81, 3),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05), bias_initializer=initializers.Zeros(), padding='valid', use_bias=True),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05), bias_initializer=initializers.Zeros()),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05), bias_initializer=initializers.Zeros()),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05), bias_initializer=initializers.Zeros()),
        Dense(3, activation='softmax', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05), bias_initializer=initializers.Zeros())]) ## 3 unit - classes
    model.compile(optimizer=optimizer, loss=loss_name, metrics=['accuracy'])

    return model

def save_model_and_preprocessing(model, mean, std, model_path): ## Save the CNN model with normalization parameters
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'mean': mean, 'std': std}, f)
    print(f"Model and normalization parameters saved to {model_path}")

def load_model_and_preprocessing(model_path): ## Load the CNN model with normalization parameters
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    mean = data['mean']
    std = data['std']
    print(f"Model and normalization parameters loaded from {model_path}")
    return model, mean, std

def predict_image_class(image_path, model, mean, std, class_names): ## Predict the class of a single image and plot the result
    img = Image.open(image_path).convert('RGB')
    img_array = (np.array(img) / 255. - mean) / std ## apply normalization to input image
    img_array = np.expand_dims(img_array, axis=0)

    t1 = time.time() ## time taken
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    probability = predictions[predicted_class]
    timee = time.time() - t1

    return class_names[predicted_class], probability, timee

def plot_prediction(image_path, predicted_class, probability): ## Plot the predicted class with input image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis()
    plt.show()

def plot_loss(history): ## Plot the training and validation loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def confussionmatrix(model, data, labels, dataset_name): ## Calculate the confusion matrix and accuracy on validation and evaluation sets
    predictions = model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)
    cm = confusion_matrix(labels, predicted_classes, normalize='true')
    cm_accuracy = np.trace(cm) / np.sum(cm)
    print(f'\n{dataset_name} Set Confusion Matrix:\n{cm}')
    print(f'{dataset_name} Set Confusion Matrix Accuracy: {cm_accuracy:.4f}')

def main(image_path=None, data_dir=None, learning_rate=None, epochs=None, batch_size=None, optimizer=None, loss=None, model_path=None):
    model = None

    train_data, train_labels, valid_data, valid_labels, eval_data, eval_labels = load_data(data_dir) ## load the data
    train_data, valid_data, eval_data, mean, std = normalize_data(train_data, valid_data, eval_data) ## normalize the data

    if os.path.exists(model_path):
        model, mean, std = load_model_and_preprocessing(model_path)
        print("Loaded model and normalization parameters from existing pickle file.")

    if model is None:
        model = build_cnn_model(learning_rate, optimizer, loss)
        history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(valid_data, valid_labels))
        plot_loss(history)
        save_model_and_preprocessing(model, mean, std, model_path)

    if valid_data.size and valid_labels.size: ## Evaluate model on validation dataset
        confussionmatrix(model, valid_data, valid_labels, 'Validation')
    if eval_data.size and eval_labels.size: ## Evaluate model on evaluation dataset
        confussionmatrix(model, eval_data, eval_labels, 'Evaluation')

    if image_path:
        class_names = ['grass', 'crop', 'other']
        predicted_class, probability, timee = predict_image_class(image_path, model, mean, std, class_names)
        print(f"\nPredicted Class: {predicted_class}")
        print(f"Time taken to process an image: {timee:.2f} seconds")
        plot_prediction(image_path, predicted_class, probability)

if __name__ == "__main__":
    flags = parse_args()
    main(flags.image_path, flags.data_dir, flags.learning_rate, flags.epochs, flags.batch_size, flags.optimizer, flags.loss, flags.model_path)


## Usage:
## python vegetation_classification_extension1.py C:\Users\cansu\OneDrive\Desktop\s67cbeya\PART_II_DATA\other\eval\bag1_other_000059_000059_0.png
