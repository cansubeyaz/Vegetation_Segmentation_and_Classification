import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import time
import pickle
import os
import argparse
from skimage.io import imread
from skimage.color import rgb2lab

def parse_args(): ## Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Vegetation Segmentation using Logistic Regression - PART I")
    parser.add_argument('image_path', type=str, help='Image path')
    parser.add_argument('--imax', type=int, default=2000, help='Maximum number of iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--logreg', type=str, default='logreg_model.pkl', help='Save and load the logistic regression model')
    return parser.parse_args()

class LogisticRegression():
    def __init__(self, lr=0.1, imax=2000): ## Initialize the model parameters
        self.lr = lr
        self.imax = imax
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def sigmoid(self, x): ## Sigmoid function for binary classification
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y): ## Normalize the input features (mean and std calculated from training set)
        X_normalized = self.normalize(X, fit=True)
        n_samples, n_features = X_normalized.shape
        self.weights = np.zeros(n_features)  # Zero initialize weights and bias ## self.weights = np.random.rand(n_features)  # Randomly initialize weights
        self.bias = 0 ## self.bias = np.random.rand() ## Randomly initialize bias

        for i in range(self.imax): ## Gradient descent optimization
            prediction = np.dot(X_normalized, self.weights) + self.bias
            y_predicted = self.sigmoid(prediction) ## predict probabilities

            dw = (1 / n_samples) * np.dot(X_normalized.T, (y_predicted - y)) ## gradient of the cost function with respect to the weight vector
            db = (1 / n_samples) * np.sum(y_predicted - y) ##  gradient of the cost function with respect to the bias

            self.weights -= self.lr * dw ## update the weight
            self.bias -= self.lr * db ## update the bias

    def predict(self, X): ## Predict class labels for input features
        X_normalized = self.normalize(X)
        linear_model = np.dot(X_normalized, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return np.where(probabilities >= 0.5, 1, 0) ## threshold - 0 if p < 0.5 else 1 for p in probs

    def predict_proba(self, X): ## Predict probabilities for input features
        X_normalized = self.normalize(X)
        linearmodel = np.dot(X_normalized, self.weights) + self.bias
        return self.sigmoid(linearmodel) ## probability [0,1]

    def normalize(self, X, fit=False): ## Normalize the input data using mean and calculated from training data
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 0.00000001) ## Adding small value to avoid division by zero

def save_model(segmentation_model, file_path): ## Save the logistic regression model with parameters
    with open(file_path, 'wb') as model_file:
        pickle.dump({
            'weights': segmentation_model.weights,
            'bias': segmentation_model.bias,
            'mean': segmentation_model.mean,
            'std': segmentation_model.std
        }, model_file)
    print(f"Model saved to {file_path}")

def load_model(file_path): ## Load the logistic regression model with parameters
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        segmentation_model = LogisticRegression()
        segmentation_model.weights = data['weights']
        segmentation_model.bias = data['bias']
        segmentation_model.mean = data['mean']
        segmentation_model.std = data['std']
    print(f"Model loaded from {file_path}")
    return segmentation_model

def load_and_process_data(file_path):## Load data from a pickle file and convert to LAB color space
    with open(file_path, 'rb') as fp:
        pos_data, neg_data = pickle.load(fp)

    pos_data = pos_data.T  ## Transpose the positive data(vegetation)
    neg_data = neg_data.T  ## Transpose the negative data(non-vegetation)
    pos_data_lab = rgb2lab(pos_data.reshape(-1, 1, 3)).reshape(-1, 3) ## Convert RGB data to LAB color space (for vegetation)
    neg_data_lab = rgb2lab(neg_data.reshape(-1, 1, 3)).reshape(-1, 3) ## Convert RGB data to LAB color space (for non-vegetation)

    return pos_data_lab, neg_data_lab

def load_dataset(directory): ## Load dataset from a directory of pickle files
    positive_data = []
    negative_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.pickle'):
            file_path = os.path.join(directory, filename)
            pos_data, neg_data = load_and_process_data(file_path)
            positive_data.append(pos_data)
            negative_data.append(neg_data)

    positive_data = np.concatenate(positive_data, axis=0)  ## Combine all positive data (vegetation 1)
    negative_data = np.concatenate(negative_data, axis=0)  ## Combine all negative data (non-vegetation 0)
    positive_label = np.ones(positive_data.shape[0])  ## creating labels for positive data(1)
    negative_label = np.zeros(negative_data.shape[0])  ## creating labels for negative data(0)
    X = np.vstack((positive_data, negative_data))  ## Feature matrix (X)
    y = np.hstack((positive_label, negative_label))  ## Label vector (y)

    return X, y

def plot_precision_recall(y_true, y_scores, title): ## Plot the precision-recall curve
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

def main(image_path):
    base_path = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(base_path, "PART_I_RGB_DATA/train") ## training data
    eval_path = os.path.join(base_path, "PART_I_RGB_DATA/eval")  ## evaluation data
    valid_path = os.path.join(base_path, "PART_I_RGB_DATA/valid") ## validation data
    logreg_model_path = flags.logreg

    if not os.path.exists(logreg_model_path):
        X_train, y_train = load_dataset(train_path) ## training input and labeled data
        segmentation_model = LogisticRegression(lr=flags.lr, imax=flags.imax)
        segmentation_model.fit(X_train, y_train)
        save_model(segmentation_model, logreg_model_path)
        print(f"Model training completed and saved to {logreg_model_path}.") ## save the logistic regression model
    else:
        segmentation_model = load_model(logreg_model_path) ## load the logistic regression model

    X_valid, y_valid = load_dataset(valid_path) ## validation input and labeled data
    y_scores_valid = segmentation_model.predict_proba(X_valid)
    plot_precision_recall(y_valid, y_scores_valid, 'Validation')

    X_eval, y_eval = load_dataset(eval_path) ## evaluation input and labeled data
    y_scores_eval = segmentation_model.predict_proba(X_eval)
    plot_precision_recall(y_eval, y_scores_eval, 'Evaluation')

    ## Process the input image for vegetation segmentation
    t1 = time.time()
    image = imread(image_path)/255.  ## Load image ## divide 255.
    image_lab = rgb2lab(image)  ## Convert to LAB color space
    image_combined = image_lab.reshape(-1, 3)  ## Flatten image for prediction
    vegetation_pred = segmentation_model.predict(image_combined) ## Predict vegetation pixels
    vegetation_percentage = (np.sum(vegetation_pred) / vegetation_pred.shape[0]) * 100 ## vegetation percentage

    vegetation_mask = vegetation_pred.reshape(image.shape[:2]) ## Create the binary segmented output image
    binary_segmented_image = np.zeros_like(vegetation_mask)
    binary_segmented_image[vegetation_mask == 1] = 255  ## White for vegetation
    binary_segmented_image[vegetation_mask == 0] = 0    ## Black for non-vegetation

    ## Visualization of Input and Segmented Output Images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis()
    plt.subplot(1, 2, 2)
    plt.imshow(binary_segmented_image, cmap='gray')
    plt.title(f'Segmented Binary Image\nVegetation Percentage: {vegetation_percentage:.2f}%')
    plt.axis()
    plt.show()
    print(f'\nPercentage of vegetation pixels: {vegetation_percentage:.2f}%')
    print(f'Time taken to process the image: {time.time() - t1:.2f} seconds\n')

if __name__ == "__main__":
    flags = parse_args()
    main(flags.image_path)

## Usage:
## python .\vegetation_segmentation_solution1.py C:\Users\cansu\OneDrive\Desktop\s67cbeya\PART_II_DATA\crop\train\bag1_crop_000062_000062_0.png
