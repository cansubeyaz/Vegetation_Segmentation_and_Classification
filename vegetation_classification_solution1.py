import os
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
import argparse
import time
import pickle

def parse_args(): ## Set up the command-line argument parser
    parser = argparse.ArgumentParser(description='Vegetation Binary Classification using CNN')
    parser.add_argument('image_path', type=str, nargs='?', default=None, help='Path to the image file for prediction')
    parser.add_argument('--data_dir', type=str, help='Directory containing the data', default='PART_II_DATA')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer', default=0.00001)
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=125) ## also tested on 400 epochs
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=40)
    parser.add_argument('--optimizer', type=str, help='Optimizer to use for training', choices=['adam', 'sgd', 'rmsprop'], default='adam')
    parser.add_argument('--loss', type=str, help='Loss function to use', default='binary_crossentropy')
    parser.add_argument('--train', action='store_true', help='Flag to initiate training process')
    parser.add_argument('--model_path', type=str, default='vegetation_cnn_model.pkl', help='Path to save/load the model')
    return parser.parse_args()

def load(folder, target_shape=(81, 81, 3)): ## Load images from a folder and preprocess them
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img) / 255.  # Normalize the images
            if img_array.shape == target_shape:
                images.append(img_array)
            else:
                print(f"Image {filename} due to shape mismatch: {img_array.shape}")
    return np.array(images)

def load_data(data_dir): ## Load data from directories
    class_labels = {'crop': 1, 'grass': 0, 'other': 0} ## Binary classification
    train_data, train_labels = [], []
    valid_data, valid_labels = [], []
    eval_data, eval_labels = [], []

    for cls, label in class_labels.items():
        train_folder = os.path.join(data_dir, cls, 'train')
        valid_folder = os.path.join(data_dir, cls, 'valid')
        eval_folder = os.path.join(data_dir, cls, 'eval')

        train_images = load(train_folder) ## Load the training images
        valid_images = load(valid_folder) ## Load the validation images
        eval_images = load(eval_folder) ## Load the evaluation images

        train_data.extend(train_images)
        train_labels.extend([label] * len(train_images)) ## training label data (binary)
        valid_data.extend(valid_images)
        valid_labels.extend([label] * len(valid_images)) ## validation label data (binary)
        eval_data.extend(eval_images)
        eval_labels.extend([label] * len(eval_images))  ## evaluation label data (binary)
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

def normalize_data(train_data, valid_data, eval_data): ## Mean and std normalization
    mean, std = train_data.mean(), train_data.std() ## Normalization: Mean and std calculated from training data
    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    eval_data = (eval_data - mean) / std
    return train_data, valid_data, eval_data, mean, std

def build_cnn_model(learning_rate, optimizer_name, loss_name):  ## CNN model with random initialization
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(81, 81, 3),kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05),bias_initializer=initializers.Zeros(), padding='valid', use_bias=True),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05),bias_initializer=initializers.Zeros()),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05),bias_initializer=initializers.Zeros()),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05),bias_initializer=initializers.Zeros()),
        Dense(1, activation='sigmoid', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05),bias_initializer=initializers.Zeros())])  ## 1, binary classification
    model.compile(optimizer=optimizer, loss=loss_name, metrics=['accuracy'])

    return model

'''  ## HeNormal and GlorotUniform(Xavier) weight initialization
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(81, 81, 3), kernel_initializer=HeNormal(), bias_initializer='zeros', padding = 'valid'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal(), bias_initializer='zeros'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer=HeNormal(), bias_initializer='zeros'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=HeNormal(), bias_initializer='zeros'),
        Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(), bias_initializer='zeros')])
    model.compile(optimizer=optimizer, loss=loss_name, metrics=['accuracy'])

    return model
'''

def save_model(model, file_path, mean, std): ## Save the CNN model with normalization parameters
    with open(file_path, 'wb') as model_file:
        pickle.dump({'model': model, 'mean': mean, 'std': std}, model_file)
    print(f"Model and normalization parameters saved to {file_path}")

def load_model(file_path): ## Load the CNN model with normalization parameters
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        mean = data['mean']
        std = data['std']
    print(f"Model and normalization parameters loaded from {file_path}")
    return model, mean, std

def plot_precision_recall(y_true, y_scores, title=''): ## Plot the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 0.00000001)
    f1 = np.argmax(f1_scores)

    plt.plot(recall, precision)
    plt.scatter([recall[f1]], [precision[f1]], color='red', marker='*', label=f'F1 Score = {f1_scores[f1]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {title}')
    plt.legend(loc='lower left')
    plt.show()
    print(f'\n---{title} Set:')
    print(f'Precision: {precision[f1]:.4f}')
    print(f'Recall: {recall[f1]:.4f}')
    print(f'F1 Score: {f1_scores[f1]:.4f}')
    print(f'Threshold: {thresholds[f1]:.4f}')

def predict_and_plot_image_class(image_path, model, mean, std, class_names): ## Predict the class of a single image and plot the result
    img = Image.open(image_path).convert('RGB')
    img_array = (np.array(img) / 255. - mean) / std ## normalization
    img_array = np.expand_dims(img_array, axis=0)

    t1 = time.time() ## time taken
    prediction = model.predict(img_array)[0, 0]
    timee = time.time() - t1

    predicted_class = class_names[int(prediction >= 0.5)] ## apply class label
    plt.figure()
    plt.imshow(np.array(img) / 255.) ## Display the original image
    plt.title(f"Predicted Class: {predicted_class} \nPredicted Probability: {prediction:.2f}")
    plt.axis()
    plt.show()
    print(f"\nPredicted Class: {predicted_class}")
    print(f"Predicted Probability: {prediction:.2f}")
    print(f"Time taken to process an image: {timee:.2f} seconds")

def main(image_path=None, data_dir=None, learning_rate=None, epochs=None, batch_size=None, optimizer=None, loss=None, train=False, model_path=None):
    model_exists = os.path.exists(model_path)
    if not train and not model_exists:
        print("Training the CNN model.")
        train = True

    if train: ## Load and normalize data
        train_data, train_labels, valid_data, valid_labels, eval_data, eval_labels = load_data(data_dir)
        train_data, valid_data, eval_data, mean, std = normalize_data(train_data, valid_data, eval_data)

        model = build_cnn_model(learning_rate, optimizer, loss) ## Build and train the model
        history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs,validation_data=(valid_data, valid_labels))

        plt.figure() ## Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        save_model(model, model_path, mean, std) ## Save the trained model and normalization parameters

        for dataset, name in zip([(valid_data, valid_labels), (eval_data, eval_labels)], ['Validation', 'Evaluation']): ## Plot the pr curves on validation and evaluation sets
            X, y = dataset
            y_pred = model.predict(X).ravel()
            plot_precision_recall(y, y_pred, title=name + ' Set')
    else: ## Load the pre-trained model
        model, mean, std = load_model(model_path)

    if image_path: ## Predict and plot for a specific image if provided
        class_names = ['weed', 'crop']
        predict_and_plot_image_class(image_path, model, mean, std, class_names)

if __name__ == "__main__":
    flags = parse_args()
    main(flags.image_path, flags.data_dir, flags.learning_rate, flags.epochs, flags.batch_size, flags.optimizer,
         flags.loss, flags.train, flags.model_path)

## Usage:
## python vegetation_classification_solution1.py C:\Users\cansu\OneDrive\Desktop\s67cbeya\PART_II_DATA\other\eval\bag1_other_000059_000059_0.png
