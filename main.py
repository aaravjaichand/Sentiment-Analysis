import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

categorizations = ["This review has a positive tone", "This review has a negative tone"]

def get_data():
    saved_file = Path("data.npz")
    if saved_file.exists():
        data = np.load(saved_file)
        inputs, targets = data["inputs"], data["targets"]
    else:
        dataset = pd.read_csv("dataset.csv")
        inputs = dataset['review_text'].tolist() 
        targets = dataset['label'].tolist()  
        np.savez(saved_file, inputs=inputs, targets=targets)
    return inputs, targets

def preprocess_data(inputs):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(inputs)
    return X, vectorizer

def train_sklearn_model(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_sklearn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"sklearn Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

def evaluate_transformers_model(model, tokenizer, inputs, targets):
    with torch.inference_mode():
        predictions = []
        for input_text in tqdm(inputs, desc="Evaluating"):
            tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
            output = model(**tokens)
            pred = torch.argmax(output.logits, dim=1).item()
            predictions.append(pred)
        
    accuracy = np.mean(np.array(predictions) == np.array(targets))
    print(f"Transformers Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(targets, predictions))

def main():
    inputs, targets = get_data()
    

    X, vectorizer = preprocess_data(inputs)
    y = np.array(targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    sklearn_model = train_sklearn_model(X_train, y_train)
    evaluate_sklearn_model(sklearn_model, X_test, y_test)
    
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    transformers_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    evaluate_transformers_model(transformers_model, tokenizer, inputs, targets)

if __name__ == "__main__":
    main()