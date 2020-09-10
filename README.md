# Comparison-of-Hybrid-Neural-Network-Methodologies-for-Sentiment-Emotion-Analysis

# Masters in Data Analytics Project

## Project: Comparison of Hybrid Neural Network Methodologies for Sentiment & Emotion Analysis

## Table of Contents

- [Overview](#overview)
- [Methodology](#method)
- [Components](#components)
  - [Data Extraction and Pre-Processing](#data)
  - [Emotion Classification using NRC Lexicon and LSTM based DNN](#emotionnrclstm)
  - [Emotion Classification using Vader Lexicon and LSTM+CNN based DNN](#emotionvaderlstmcnn)
  - [Sentiment Classification using Vader Lexicon and LSTM+CNN based DNN](#sentimentvaderlstmcnn)
  - [Sentiment Polarity Analysis using Vader Lexicon and Bi-Directional LSTM based DNN](#sentimentvaderbilstm)
- [Running the Code](#running)
- [Screenshots](#screenshots)
- [System Configuration steps](#config)
- [File Descriptions](#files)
- [Credits and Acknowledgements](#credits)

***

<a id='overview'></a>

### Overview
Twitter tweets play an important role in every organisation. This project is based on analysing the English tweets and categorizing the tweets based on the sentiment and emotions of the user. The literature survey conducted showed promising results of using hybrid methodologies for sentiment and emotion analysis. Four different hybrid methodologies have been used for analysing the tweets belonging to various categories. A combination of classification and regression approaches using different deep learning models such as Bidirectional LSTM, LSTM and Convolutional neural network (CNN) are implemented to perform sentiment and behaviour analysis of the tweets. A novel approach of combining Vader and NRC lexicon is used to generate the sentiment and emotion polarity and categories. The evaluation metrics such as accuracy, mean absolute error and mean square error are used to test the performance of the model.  The business use cases for the models applied here can be to understand the opinion of customers towards their business to improve their service. Contradictory to the suggestions of Google’s S/W ratio method, LSTM models performed better than using CNN models for categorical as well as regression problems.

<a id='method'></a>

### Methodology

The below diagram shows the methodology followed for the project and the analysis therein:

![Screenshot1](/images/method.png)

<a id='components'></a>

### Components

<a id='data'></a>

#### Data Extraction and Pre-Processing
File _'Data Cleaning and Pre-Processing.ipynb'_ :

- Imports the full dataset containing twitter tweets for 1 day (01-Aug-2019)
- Filters the data using Language, Retweets and Hashtags
- Exports the filtered and fina data into a .csv file

<a id='emotionnrclstm'></a>

#### Emotion Classification using NRC Lexicon and LSTM based DNN
File _'NRC_Emotion Category.ipynb'_ :

- Imports the filtered and final data of twitter tweets
- Performs text analysis on the data
- Applies NRC Lexicon to generate the emotions for each tweet
- Applies the LSTM based DNN to create a model that predicts the emotion based on the tweet
- Generates evaluation metrics for comparison

<a id='emotionvaderlstmcnn'></a>

#### Emotion Classification using Vader Lexicon and LSTM+CNN based DNN
File _'Vader_Emotion Category.ipynb'_ :

- Imports the filtered and final data of twitter tweets
- Performs text analysis on the data
- Applies Vader Lexicon along with clustering to generate the emotions for each tweet
- Applies the LSTM and CNN based DNN to create a model that predicts the emotion based on the tweet
- Generates evaluation metrics for comparison

<a id='sentimentvaderlstmcnn'></a>

#### Sentiment Classification using Vader Lexicon and LSTM+CNN based DNN
File _'Sentiment Category.ipynb'_ :

- Imports the filtered and final data of twitter tweets
- Performs text analysis on the data
- Applies Vader Lexicon to generate the sentiment for each tweet
- Applies the LSTM and CNN based DNN to create a model that predicts the sentiment based on the tweet
- Generates evaluation metrics for comparison

<a id='sentimentvaderbilstm'></a>

#### Sentiment Polarity Analysis using Vader Lexicon and Bi-Directional LSTM based DNN
File _'Sentiment Polarity.ipynb'_ :

- Imports the filtered and final data of twitter tweets
- Performs text analysis on the data
- Applies Vader Lexicon to generate the sentiment polarity scores for each tweet
- Applies the Bi-Directional LSTM based DNN to create a model that predicts the sentiment polarity based on the tweet
- Generates evaluation metrics for comparison

<a id='running'></a>

### Running the Code

Download the base dataset from the below link and store it in the same folder as the codes - 
https://archive.org/details/twitterstream?and[]=year\%3A"2019"

(Only download the 01-Aug-2019 data zip file)

1) Execute the "Data Cleaning and Pre-processing.ipynb" file to generate the final dataset used for analysis

2) Execute the respective model ipynb files to perform the analysis and see the results.

<a id='screenshots'></a>

### Screenshots

![Screenshot2](/images/isw_loss.png)
![Screenshot3](/images/isw_mae.png)
![Screenshot4](/images/ram_accuracy.png)
![Screenshot5](/images/ram_conf_matr.png)
![Screenshot6](/images/ram_loss.png)
![Screenshot7](/images/ram_postcls.png)
![Screenshot8](/images/ram_precls.png)
![Screenshot9](/images/san_accuracy.png)
![Screenshot10](/images/san_cat_post.png)
![Screenshot11](/images/san_cat_pre.png)
![Screenshot12](/images/san_conf.png)
![Screenshot13](/images/san_wrdcld.png)
![Screenshot14](/images/shr_accuracy.png)
![Screenshot15](/images/shr_conf.png)
![Screenshot16](/images/shr_loss.png)

<a id='config'></a>

### System Configuration Steps

In order to run the code, below are the necessary requirements:

- Python and Jupyter Notebook: As the code for data extraction and merging is written in Python, Python along with Jupyter Notebook as IDE is required for the execution of the same. Below are the packages that are required as part of the pre-requisites for the same:

os, tarfile, pandas, pyspark, vaderSentiment, matplotlib, numpy, re, tensorflow, sklearn, bs4, string, nltk, emoji, nrclex, seaborn, keras, itertools, scikitplot, gensim, operator, pickle, pathlib, nlp_utils

<a id='files'></a>

### File Descriptions

Below are the files and the folders that are part of the project implementation:

1. Cleaned Data:
- August01_Tweets_Final.csv: Contains the data used for analysis after filtering the raw tweets.

2. Code:
- Data Cleaning and Pre-Processing.ipynb: Contains the code to clean, pre-process and filter the raw twitter tweets data
- NRC_Emotion Category.ipynb: Contains to code to apply Emotion Classification using NRC Lexicon and LSTM based DNN model
- Sentiment Category.ipynb: Contains the code to apply Sentiment Classification using Vader Lexicon and LSTM+CNN based DNN model
- Sentiment Polarity.ipynb: Contains the code to apply Sentiment Polarity Analysis using Vader Lexicon and Bi-Directional LSTM based DNN model
- Vader_Emotion Category.ipynb: Contains the code to apply Emotion Classification using Vader Lexicon and LSTM+CNN based DNN model

### Credits and Acknowledgements

* [Archive Team: The Twitter Stream Grab](https://archive.org/details/twitterstream?and[]=year\%3A%222019%22) for providing the data used for this project.
* [NCI](https://www.ncirl.ie/) for a challenging project as part of their full-time masters in data analytics course subject 'Data Mining and Machine Learning 2'
