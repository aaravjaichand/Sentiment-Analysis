import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

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


def singleUse(review, tokenizer, model):
    with torch.inference_mode():
        input_text = review
        tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        output = model(**tokens)
        
        pred = torch.argmax(output.logits, dim=1).item()

        return pred
    
