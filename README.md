# Sentiment-Analysis

## Setup Instructions:
- Install [Python](https://www.python.org) version 3.8 or above

#### Setup Python Environment:

- Install Miniconda or Anaconda (If not already):
  - Download and install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) (if not already).
  - Follow the installation instructions for your operating system.
  
- After installing Conda, create a new environment from `environment.yml` using `conda env create -f environment.yml`
- Follow the instructions in your terminal carefully

(If using visual studio code, make sure that you have the environment selected when using `>Python: Select Interpreter')

That's it for setup.

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


## Distilbert Model Accuracy and Performance Metrics:

**Model Accuracy:** 92.40%



## References:
Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020, February 29). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. ArXiv.org. [https://doi.org/10.48550/arXiv.1910.01108)](https://doi.org/10.48550/arXiv.1910.01108)
‌


Dataset: [https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis](https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis)


## Profile:

See my profile for more repositories like this [here](https://github.com/aaravjaichand)



# Enjoy!
