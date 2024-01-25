# Fake-News-Prediction-system-utilizing-Natural-Language-Processing-NLP-and-Machine-Learning
This project revolves around the development of a Fake News Prediction system utilizing NLP and ML techniques. The primary objective is to discern between real and fake news articles based on their textual content. 

## Project Overview
This project revolves around the development of a Fake News Prediction system utilizing Natural Language Processing (NLP) and Machine Learning techniques. The primary objective is to discern between real and fake news articles based on their textual content. The dataset employed in this project includes various features such as the title, author, text, and a label indicating whether the news is real or fake (0 for real news, 1 for fake news).

## About the Dataset
The dataset contains the following columns:

id: A unique identifier for each news article.
title: The title of the news article.
author: The author of the news article.
text: The textual content of the article (which may be incomplete).
label: A binary label indicating whether the news article is real (0) or fake (1).
## Dependencies
Before running the project, ensure you have the following Python libraries installed:

NumPy: A library for numerical operations.
Pandas: A data manipulation and analysis library.
NLTK (Natural Language Toolkit): Used for text processing tasks.
Scikit-Learn: A machine learning library for model training and evaluation.
Matplotlib: A plotting library for visualizations.
You can install these libraries using the following command:


pip install numpy pandas nltk scikit-learn matplotlib
## Data Pre-processing
The data pre-processing steps include:

Loading the dataset into a Pandas DataFrame.
Handling missing values by replacing them with empty strings.
Merging the 'author' and 'title' columns into a new column called 'content'.
Performing text pre-processing, including stemming, converting to lowercase, and removing stopwords.
## Feature Extraction
The textual data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
Feature extraction is the process of transforming raw data into a format that a machine learning model can understand. In natural language processing (NLP) tasks like text classification (as in this project), converting text data into numerical features is crucial. Here, TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is employed.
## Model Training
The dataset is split into training and test sets, and a Logistic Regression model is chosen for training. The model is then trained on the training set.
Model training involves using a machine learning algorithm to learn patterns and relationships within the data. In this project, a Logistic Regression model is chosen.

### Logistic Regression:
Logistic Regression: Despite its name, logistic regression is used for binary classification problems. It models the probability that an instance belongs to a particular class. The logistic function (sigmoid) is applied to a linear combination of input features.
## Evaluation
The accuracy score of the model is evaluated on both the training and test datasets.
Evaluation assesses the performance of the model on unseen data (test set). Several metrics can be used, but in this project, accuracy is employed.

### Accuracy:
Accuracy: It is the ratio of correctly predicted instances to the total instances. It is a common metric for classification problems.
## Making Predictions
A sample news article is selected from the test set, and the trained model predicts whether the news is real or fake. The actual label of the news is then compared with the predicted label.

## Results
The accuracy score on the training data is approximately 98.66%.
The accuracy score on the test data is approximately 97.91%.


### Follow the instructions in the notebook to execute code cells and analyze the results.
