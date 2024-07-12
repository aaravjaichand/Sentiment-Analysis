import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel

categorizations = ["This review has a postive tone", "This review has a negative tone"]

def getData():

    saved_file = Path("data.npz")

    if saved_file.exists():
        line = np.load(saved_file)

        inputs, targets = line["inputs"], line["targets"]

    else:
        dataset = pd.read_csv("dataset.csv") 

        lines = 1000

        inputs, targets = [], []

        for i in tqdm(range(lines), desc="Extracting data from .csv"):
            line = dataset.iloc[i]
            currReview = line.iloc[1]
            target = line.iloc[2]

            inputs.append(currReview)
            targets.append(target)
        
        np.savez(saved_file, inputs = inputs, targets = targets)
        
    return inputs, targets


def sentence_transformers():
    inputs, targets = getData()

    inputs, targets = inputs.tolist(), targets.tolist()

    def batches(layer):
        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            last_hidden_states = model(**tokenizer(inputs, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states
            last_hidden_states_labels = model(**tokenizer(categorizations, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states

        predictions = (last_hidden_states_labels[layer].mean(axis=1) @ last_hidden_states[layer].mean(axis=1).T).softmax(0).argmax(axis=0)
        prediction_strings = np.array(categorizations)[np.array(predictions)]


        # Calculate accuracy
        correct_predictions = (predictions == torch.tensor(targets)).sum().item()
        total_predictions = len(targets)
        accuracy = correct_predictions / total_predictions

        print(f"Accuracy: {accuracy * 100:.2f}%")

    def graph():
        listAccuracies = []

        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            last_hidden_states = model(**tokenizer(inputs, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states
            last_hidden_states_labels = model(**tokenizer(categorizations, return_tensors='pt', padding=True, truncation=True), output_hidden_states=True).hidden_states

        bestLayer = -1
        bestAcc = -1

        for i in range(6):
            predictions = (last_hidden_states_labels[i].mean(axis=1) @ last_hidden_states[i].mean(axis=1).T).softmax(0).argmax(axis=0)
            prediction_strings = np.array(categorizations)[np.array(predictions)]
            correct_predictions = (predictions == torch.tensor(targets)).sum().item()
            total_predictions = len(targets)
            accuracy = correct_predictions / total_predictions
            listAccuracies.append(accuracy)

            if accuracy > bestAcc:
                bestAcc = accuracy
                bestLayer = i




        plt.xlabel("Layer used by model")
        plt.ylabel("Accuracy")
        plt.plot(listAccuracies)   
        plt.show()

        return bestLayer, bestAcc
    
    funcToUse = int(input("Would you like to find the optimal layer(1) to use or run on one layer(2)? My choice: "))
    
    if funcToUse == 2:
        batches(int(input("What layer (0-6): ")))
    else:
        bestLayer, bestAcc = graph()
        print("Optimal layer:", bestLayer)
        print("Accuracy with optimal layer: ", bestAcc)

sentence_transformers()