# Sentiment Analysis


## Setup Instructions:
- Install [Python](https://www.python.org) version 3.8 or above

#### Setup Python Environment:

- Install Conda using `pip install conda`
- Install dependencies from the `environment.yml` file using `conda env create -f environment.yml`. This creates an environment called `sentiment_analysis_environment`.
- Activate the environment using `conda activate sentiment_analysis_environment` or by going into settings for your code editor and selecting `sentiment_analysis_environment` from there, if your code editor has that feature. 


#### Dataset:

- Download the dataset [here](https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis). It's the one I trained and tested with, but feel free to choose any other one, make sure that it follows the following criteria. **Please note that this dataset's targets must be binary(1's and 0's).**  
- Adding the dataset in the code is simple. Download it into the same folder as `main.py` and follow the setup instructions when running main.py.


## Models:

#### Models Used:
- sklearn models traiend on [dataset](https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis)
- distilbert fine tuned model by HuggingFace, available [here](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

#### Model Accuracy Trained and Tested on 1000 reviews from this [dataset](https://www.kaggle.com/datasets/mohidabdulrehman/vs-sentiment-analysis):

