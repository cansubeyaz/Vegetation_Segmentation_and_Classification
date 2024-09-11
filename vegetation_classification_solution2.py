import os
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from PIL import Image
import time
import pickle
from functools import wraps

def parse_args():  ## Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Vegetation Classification using Custom LBP and SVC")
    parser.add_argument('image_path', type=str, help="Image path")
    parser.add_argument('--radius', type=int, default=2, help="Radius for LBP features")
    parser.add_argument('--n_points', type=int, default=16, help="Number of points for LBP features") ##10
    parser.add_argument('--C', type=float, default=1.0, help="Regularization parameter for the SVC")
    parser.add_argument('--model_path', type=str, help="Save/load the model", default='svc_classifier.pkl')
    return parser.parse_args()

def timing_decorator(func): ## Decorator to measure the execution time
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        timee = t2 - t1
        return result, timee
    return wrapper

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

class LBPFeatureExtractor: ## Local Binary Pattern for feature extraction
    def __init__(self, radius, n_points): ##Initialize the LBP feature extractor with radius and number of points
        self.radius = radius
        self.n_points = n_points

    def extract(self, image): ## To extract LBP features from an image
        return self.lbp(image)

    def lbp(self, image):
        height, width, _ = image.shape
        lbp_image = np.zeros((height-2*self.radius, width-2*self.radius), dtype=np.uint8)
        for i in range(self.radius, height-self.radius): ## Process each pixel with respect to its neighborhood
            for j in range(self.radius, width-self.radius):
                center = image[i, j]
                binary = ''
                for n in range(self.n_points): ## Compare the center pixel with its circular neighborhood
                    theta = (2 * np.pi * n) / self.n_points  ## Angle to sample points in the neighborhood
                    x_neighbor = int(i + self.radius * np.sin(theta)) ## x coordinate of neighbor
                    y_neighbor = int(j + self.radius * np.cos(theta)) ## y coordinate of neighbor
                    if image[x_neighbor, y_neighbor].mean() >= center.mean(): ## Threshold and build a binary string
                        binary += '1'
                    else:
                        binary += '0'
                lbp_image[i-self.radius, j-self.radius] = int(binary, 2) ## Convert binary string to decimal
        (hist, _) = np.histogram(lbp_image.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2)) ## To compute histogram of the LBP values
        hist = hist.astype("float")
        hist /= (hist.sum() + 0.000001) ## Normalize the histogram

        return hist

class SVClassifier: ## Support Vector Machine (SVM)
    def __init__(self, radius, n_points, C): ## Initialize the SVM classifier with LBP feature extraction.
        self.model = SVC(probability=True, C=C)
        self.feature_extractor = LBPFeatureExtractor(radius, n_points) ## Set up the LBP feature extractor.

    def train(self, train_data, train_labels): ## Train the SVM classifier using the extracted features from training data
        features = [self.feature_extractor.extract(image) for image in train_data] ## Extract features for each image in the training dataset
        self.model.fit(features, train_labels) ## Train the SVM model on the extracted features

    @timing_decorator
    def predict_with_probability(self, image):
        feature = self.feature_extractor.extract(image) ## Extract features using the LBP feature extractor
        probabilities = self.model.predict_proba([feature])[0] ## Predict probability
        predicted_class = self.model.predict([feature])[0] ## Predict class
        return predicted_class, probabilities

    def plot_precision_recall(self, data, labels, title): ## Plot the precision-recall curve
        features = [self.feature_extractor.extract(image) for image in data]
        probabilities = self.model.predict_proba(features)[:, 1]
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        f1_scores = 2 * (precision * recall) / (precision + recall + 0.00000001)
        f1 = np.argmax(f1_scores)
        plt.figure()
        plt.plot(recall, precision)
        plt.scatter(recall[f1], precision[f1], color='red', marker='*', label=f'F1 Score: {f1_scores[f1]:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {title}')
        plt.legend(loc='lower left')
        plt.show()
        print(f"\n---{title} Set:")
        print(f"Precision: {precision[f1]:.4f}")
        print(f"Recall: {recall[f1]:.4f}")
        print(f"F1 Score: {f1_scores[f1]:.4f}")
        print(f"Threshold: {thresholds[f1]:.4f}")

def save_model(classifier, file_path, mean, std): ## Save the model with normalization parameters
    with open(file_path, 'wb') as model_file:
        pickle.dump({'model': classifier.model, 'feature_extractor': classifier.feature_extractor, 'mean': mean, 'std': std}, model_file)
    print(f"Model and normalization parameters saved to {file_path}")

def load_model(file_path): ## Load the model with normalization parameters
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        feature_extractor = data['feature_extractor']
        mean = data['mean']
        std = data['std']
    return model, feature_extractor, mean, std

def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.
    return img

def visualize_prediction(image_path, prediction, probability):
    img_rgb = Image.open(image_path)
    plt.imshow(img_rgb)
    plt.title(f'Predicted Class: {"Crop" if prediction == 1 else "Weed"}')
    plt.axis()
    plt.show()

def main():
    args = parse_args()
    data_dir = 'PART_II_DATA'
    train_data, train_labels, valid_data, valid_labels, eval_data, eval_labels = load_data(data_dir) ## Load the data
    train_data, valid_data, eval_data, mean, std = normalize_data(train_data, valid_data, eval_data) ## Normalize the data

    if os.path.exists(args.model_path): ## Load the model
        model, feature_extractor, mean, std = load_model(args.model_path)
        classifier = SVClassifier(args.radius, args.n_points, args.C)
        classifier.model = model
        classifier.feature_extractor = feature_extractor
        print(f"Model loaded from {args.model_path}")
    else: ## Save the model
        classifier = SVClassifier(args.radius, args.n_points, args.C)
        classifier.train(train_data, train_labels)
        if args.model_path:
            save_model(classifier, args.model_path, mean, std)
            print(f"Model saved to {args.model_path}")

    img = process_image(args.image_path)
    classifier.plot_precision_recall(valid_data, valid_labels, 'Validation') ## plot pr curve on validation dataset
    classifier.plot_precision_recall(eval_data, eval_labels, 'Evaluation') ## plot pr curve on evaluation dataset

    (prediction, probabilities), timee = classifier.predict_with_probability(img)
    visualize_prediction(args.image_path, prediction, probabilities)

    print(f"Time taken to process an image: {timee:.4f} seconds")

if __name__ == "__main__":
    main()

## Usage:
## python vegetation_classification_solution2.py C:\Users\cansu\OneDrive\Desktop\project_lecture\PART_II_DATA\other\eval\bag1_other_000059_000059_0.png
