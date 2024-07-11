import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


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

inputs, targets = getData()

print(inputs[123], targets[123])