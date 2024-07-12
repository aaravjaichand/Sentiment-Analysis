import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from models.sklearn_model import preprocess_data, train_sklearn_model, evaluate_sklearn_model
from models.transformers_model import evaluate_transformers_model


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


def main():
    inputs, targets = get_data()
    

    # userChoice = int(input("Test on singular review? "))







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