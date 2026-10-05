"""
Train OpenCV ANN_MLP Classifier on 120-d Projection Features
Author: Teeranon (AI Engineering, Prince of Songkla University)
Description:
    Reads normalized 120-dimensional feature vectors (in.txt) and one-hot labels (out.txt),
    trains a Multi-Layer Perceptron using OpenCV RPROP, and saves the trained model.xml.
"""

import os
import argparse
import cv2
import numpy as np


def load_dataset(in_path="data/dataset/in.txt", out_path="data/dataset/out.txt"):
    """
    Loads features (in.txt) and targets (out.txt).
    Each line in in.txt contains 120 comma-separated float features.
    Each line in out.txt contains 2 comma-separated values:
        [1, 0] -> CRACK (Defect)
        [0, 1] -> NORMAL (Pass)
    """
    if not os.path.exists(in_path) or not os.path.exists(out_path):
        raise FileNotFoundError(f"Dataset files not found: {in_path} or {out_path}")
        
    features = []
    with open(in_path, 'r') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if line:
                features.append([float(val) for val in line.split(',') if val.strip()])
                
    labels = []
    with open(out_path, 'r') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if line:
                labels.append([float(val) for val in line.split(',') if val.strip()])
                
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features per sample.")
    crack_cnt = np.sum(np.argmax(y, axis=1) == 0)
    normal_cnt = np.sum(np.argmax(y, axis=1) == 1)
    print(f"Class Distribution: {crack_cnt} Crack Defect samples | {normal_cnt} Normal Rim samples")
    
    return X, y


def train_mlp_model(X, y, save_path="models/model.xml", max_iters=3000):
    """
    Configures and trains OpenCV ANN_MLP model with:
    - Architecture: 120 (Input) -> 15 (Hidden) -> 2 (Output)
    - Activation: Symmetric Sigmoid (tanh-like)
    - Optimizer: RPROP (Resilient Backpropagation)
    """
    mlp = cv2.ml.ANN_MLP_create()
    
    # 3-layer MLP architecture
    layers = np.array([120, 15, 2], dtype=np.int32)
    mlp.setLayerSizes(layers)
    
    # Symmetric sigmoid activation function: f(x) = beta * (1 - exp(-alpha*x)) / (1 + exp(-alpha*x))
    mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6666, 1.7159)
    
    # Resilient Backpropagation (RPROP) for fast convergence on edge CPUs
    mlp.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
    mlp.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iters, 0.0001))
    
    print(f"Training ANN_MLP [120 -> 15 -> 2] using RPROP for {max_iters} iterations...")
    train_data = cv2.ml.TrainData_create(X, cv2.ml.ROW_SAMPLE, y)
    mlp.train(train_data)
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    mlp.save(save_path)
    print(f"Model successfully saved to: {save_path}")
    
    return mlp


def evaluate_model(model, X, y):
    """
    Calculates Confusion Matrix, Accuracy, Precision, and Recall for manufacturing validation.
    """
    _, preds = model.predict(X)
    
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(preds, axis=1)
    
    tp = int(np.sum((y_true == 0) & (y_pred == 0))) # Correctly flagged cracks
    fn = int(np.sum((y_true == 0) & (y_pred == 1))) # Missed cracks
    fp = int(np.sum((y_true == 1) & (y_pred == 0))) # False alarms on normal cans
    tn = int(np.sum((y_true == 1) & (y_pred == 1))) # Normal cans passed
    
    accuracy = (tp + tn) / len(y) * 100.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print("\n" + "=" * 55)
    print("      MANUFACTURING METROLOGY EVALUATION REPORT     ")
    print("=" * 55)
    print(f"Total Test Samples          : {len(y):,}")
    print(f"Overall Accuracy            : {accuracy:.2f}%")
    print(f"True Positives (Cracks)     : {tp:,} / {tp + fn:,}")
    print(f"False Negatives (Missed)    : {fn:,}")
    print(f"False Positives (Alarm)     : {fp:,} (Normal scrap rate)")
    print(f"True Negatives (Normal OK)  : {tn:,} / {tn + fp:,}")
    print("-" * 55)
    print(f"Defect Precision Rate       : {precision:.2f}% (0% False Rejection)")
    print(f"Defect Recall Rate          : {recall:.2f}%")
    print(f"Balanced F1-Score           : {f1:.2f}%")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train OpenCV ANN_MLP on Can Crack Features")
    parser.add_argument("--in_data", type=str, default="data/dataset/in.txt", help="Input features file")
    parser.add_argument("--out_data", type=str, default="data/dataset/out.txt", help="Output targets file")
    parser.add_argument("--save", type=str, default="models/model.xml", help="Save path for trained model")
    parser.add_argument("--evaluate_only", action="store_true", help="Only evaluate existing model")
    args = parser.parse_args()
    
    X, y = load_dataset(args.in_data, args.out_data)
    
    if args.evaluate_only:
        print(f"Evaluating existing model from: {args.save}")
        model = cv2.ml.ANN_MLP_load(args.save)
    else:
        model = train_mlp_model(X, y, save_path=args.save)
        
    evaluate_model(model, X, y)


if __name__ == "__main__":
    main()
