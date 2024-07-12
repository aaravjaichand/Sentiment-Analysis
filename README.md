# Sentiment Analysis Project
From [Aarav Jaichand](https://github.com/aaravjaichand)


## Setup Instructions:
- Install [Python](https://www.python.org) version 3.8 or above

#### Setup Python Environment:

- Install Conda using `pip install conda`
- Install dependencies from the `environment.yml` file using `conda env create -f environment.yml`. This creates an environment called `sentiment_analysis_environment`.
- Activate the environment using `conda activate sentiment_analysis_environment` or by going into settings for your code editor and selecting `sentiment_analysis_environment` from there, if your code editor has that feature. 


#### Dataset:

- Download the dataset [here](https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis). It's the one I trained and tested with, but feel free to choose any other one, make sure that it follows the following criteria. **Please note that this dataset's targets must be binary(1's and 0's).**  
- Adding the dataset in the code is simple. Download it into the same folder as `main.py` and follow the setup instructions when running main.py.


#### Running the Code:
When running `main.py`, if a dataset is not present, you are prompted to add one. To get started, please download the dataset above. 

Below is an example of how to use this program for the first time with this dataset. The other times you use it, it is pretty straightforward.


**Terminal Interaction**

> **Enter .csv file name with file extension:** `dataset.csv`

> **What is the exact name (case sensitive) of the column with the text (reviews, posts, whatever):** `sentence`

> **What is the exact name (case sensitive) of the column with the target:** `label`

> **Would you like to see accuracies being calculated on this dataset(1) OR run on a singular review(2):** `1`



## Models:

#### Models Used:
- sklearn models traiend on [dataset](https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis)
- distilbert fine tuned model by HuggingFace, available [here](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)



## Sklearn Model Accuracy and Performance Metrics:



**Model Accuracy:** 82.00%

| Metric           | Class 0 | Class 1 | Macro Average | Weighted Average |
|------------------|---------|---------|---------------|------------------|
| **Precision**    | 0.78    | 0.86    | 0.82          | 0.82             |
| **Recall**       | 0.86    | 0.78    | 0.82          | 0.82             |
| **F1-Score**     | 0.82    | 0.82    | 0.82          | 0.82             |
| **Support**      | 96      | 104     | 200           | 200              |


## Distilbert Model Accuracy and Performance Metrics:

**Model Accuracy:** 92.40%

| Metric           | Class 0 | Class 1 | Macro Average | Weighted Average |
|------------------|---------|---------|---------------|------------------|
| **Precision**    | 0.93    | 0.92    | 0.92          | 0.92             |
| **Recall**       | 0.92    | 0.93    | 0.92          | 0.92             |
| **F1-Score**     | 0.92    | 0.92    | 0.92          | 0.92             |
| **Support**      | 500     | 500     | 1000          | 1000             |



## References:
Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020, February 29). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. ArXiv.org. [https://doi.org/10.48550/arXiv.1910.01108)](https://doi.org/10.48550/arXiv.1910.01108)
‌


Dataset (No DOI, so link is here): [https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis](https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis)


