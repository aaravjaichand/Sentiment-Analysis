import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from models.sklearn_model import preprocess_data, train_sklearn_model, evaluate_sklearn_model
from models.transformers_model import evaluate_transformers_model, singleUse


categorizations = ["This review has a positive tone", "This review has a negative tone"]
categorizationsDict = {1: "Positive", 0: "Negative"}



if Path("output.out").exists():
    Path("output.out").unlink()

def get_data():
    userFile = Path("userfile.npz")
    
    if userFile.exists():
        data = np.load(userFile)

        identity = data["identity"]

        print(identity)

        string = "Do you want to use this dataset above? : "
        
        sc = input(string)

        if sc == "Yes" or sc == "yes" or sc == "y" or sc == "Y":
            inputs, targets = data["inputs"], data["targets"]
            print("File Setup Complete!")
        else:

            refresh = input("Do you want to refresh the data in this dataset?")

            if refresh == "Yes" or refresh == "yes" or refresh == "y" or refresh == "Y":
                dataset = pd.read_csv(str(identity))
                text = input("What is the exact name (case senstive) of the column with the text (reviews, posts, whatever): ")
                inputs = dataset[text].tolist() 
                colTar = input("What is the exact name (case senstive) of the column with the target: ")
                targets = dataset[colTar].tolist() 
                np.savez(userFile, inputs=inputs, targets=targets, identity=str(identity))

                print("Dataset sucessfully updated. Running code...")
                return inputs, targets


            print("Restarting File Setup.")
            if userFile.exists():
                userFile.unlink()
            while True:
                fileName = input("Enter .csv file name with file extension: ")

                if Path(fileName).exists():
                    break
                else:
                    print("File not found. Plaese make sure it is in the same folder as main.py")

        
            dataset = pd.read_csv(fileName)
            text = input("What is the exact name (case senstive) of the column with the text (reviews, posts, whatever): ")
            inputs = dataset[text].tolist() 
            colTar = input("What is the exact name (case senstive) of the column with the target: ")
            targets = dataset[colTar].tolist()  
            np.savez(userFile, inputs=inputs, targets=targets, identity=fileName)
        
    else:

        while True:
            fileName = input("Enter .csv file name with file extension: ")

            if Path(fileName).exists():
                break
            else:
                print("File not found. Plaese make sure it is in the same folder as main.py")

        
        dataset = pd.read_csv(fileName)
        text = input("What is the exact name (case senstive) of the column with the text (reviews, posts, whatever): ")
        inputs = dataset[text].tolist() 
        colTar = input("What is the exact name (case senstive) of the column with the target: ")
        targets = dataset[colTar].tolist()  
        np.savez(userFile, inputs=inputs, targets=targets, identity=fileName)

    return inputs, targets


def main():
    inputs, targets = get_data()
    

    userChoice = input("Would you like to see accuracies being calculated on this dataset(1) OR run on a singular review(2): ")

   

    if userChoice == "2":
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        transformers_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        sentenceToBeClassifed = input("Enter sentence to classify: ")

        progOutput = categorizationsDict[singleUse(sentenceToBeClassifed, tokenizer, transformers_model)]
        print(progOutput)

    if userChoice == "1":
        X, vectorizer = preprocess_data(inputs)
        y = np.array(targets)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

        sklearn_model = train_sklearn_model(X_train, y_train)
        evaluate_sklearn_model(sklearn_model, X_test, y_test)
        
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        transformers_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        evaluate_transformers_model(transformers_model, tokenizer, inputs, targets, False)


        print("Program execution complete.")

        fileOutput = evaluate_transformers_model(transformers_model, tokenizer, inputs, targets, True)

        fout = open("output.out", "w")
        for foutput in fileOutput:

            print(*foutput, file=fout)
        


if __name__ == "__main__":
    main()