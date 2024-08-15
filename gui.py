import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from sklearn.model_selection import train_test_split
from models.sklearn_model import preprocess_data, train_sklearn_model, evaluate_sklearn_model
from models.transformers_model import evaluate_transformers_model, singleUse

categorizationsDict = {1: "Positive", 0: "Negative"}

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis Tool")

        self.label = tk.Label(root, text="Choose an option:")
        self.label.pack()

        self.file_button = tk.Button(root, text="Select Data File", command=self.get_data)
        self.file_button.pack()

        self.single_review_button = tk.Button(root, text="Classify Single Review", command=self.classify_single_review)
        self.single_review_button.pack()

        self.accuracy_button = tk.Button(root, text="Evaluate Model Accuracy", command=self.evaluate_accuracy)
        self.accuracy_button.pack()

        self.output_label = tk.Label(root, text="", wraplength=400)
        self.output_label.pack()

    def get_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.inputs, self.targets = self.load_data(file_path)
            self.output_label.config(text="File loaded and processed.")

    def load_data(self, file_path):
        dataset = pd.read_csv(file_path)
        text_col = simpledialog.askstring("Input", "Enter the column name for the text (reviews, posts, etc.):")
        target_col = simpledialog.askstring("Input", "Enter the column name for the target:")
        inputs = dataset[text_col].tolist()
        targets = dataset[target_col].tolist()
        np.savez("userfile.npz", inputs=inputs, targets=targets, identity=file_path)
        return inputs, targets

    def classify_single_review(self):
        # if not hasattr(self, 'inputs'):
        #     messagebox.showerror("Error", "No dataset loaded.")
        #     return
        sentence = simpledialog.askstring("Input", "Enter the sentence to classify:")
        if sentence:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            transformers_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            classification = singleUse(sentence, tokenizer, transformers_model)
            result = categorizationsDict[classification]
            self.output_label.config(text=f"Classification: {result}")

    def evaluate_accuracy(self):
        self.output_label.config(text="Test")
        if not hasattr(self, 'inputs'):
            messagebox.showerror("Error", "No dataset loaded.")
            return
        
        
        

        X, vectorizer = preprocess_data(self.inputs)
        y = np.array(self.targets)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sklearn_model = train_sklearn_model(X_train, y_train)
        sklearn_accuracy = evaluate_sklearn_model(sklearn_model, X_test, y_test)

        
 
        sklearn_accuracy = str(sklearn_accuracy)


        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        transformers_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        transformers_accuracy = evaluate_transformers_model(transformers_model, tokenizer, self.inputs, self.targets, False)

        transformers_accuracy = str(transformers_accuracy * 100)


        self.output_label.config(text="SK Learn Model Accuracy: " + sklearn_accuracy + "%; Transformers accuracy: " + transformers_accuracy + "%")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()