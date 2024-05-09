# Text_Analytics_Hate_Speech_Detection

Programming Language: Python
Purpose and Functionality:
The code file performs hate speech detection using two different machine learning models: Logistic Regression (Model 1) and Random Forest Classifier (Model 2). The main functionalities include:

Data preprocessing:

Reading and splitting the dataset into training, validation, and testing sets.
Cleaning the text data by removing special characters, converting to lowercase, and removing stop words.
Applying lemmatization to the text data.


Data visualization:

Visualizing the class distribution of the datasets using bar plots and pie charts.
Generating word clouds to visualize the most frequent words in the dataset.


Model training and evaluation:

Training Model 1 (Logistic Regression) and Model 2 (Random Forest Classifier) using different proportions of the training data (25%, 50%, 75%, and 100%).
Evaluating the models' performance on the training, validation, and testing sets using metrics such as accuracy, precision, recall, and F1-score.
Saving the trained models and vectorizers for future use.


Testing and output generation:

Loading the saved models and vectorizers.
Testing the models on the testing set and generating output files with predicted labels.


Graph generation:

Plotting graphs to visualize the performance metrics (precision, accuracy, recall, F1-score) for different data sizes and models.



Key Algorithms and Logic:

Logistic Regression (Model 1): A linear classification algorithm used for binary classification tasks.
Random Forest Classifier (Model 2): An ensemble learning algorithm that combines multiple decision trees to make predictions.
TF-IDF Vectorizer: Used to convert text data into numerical feature vectors based on the frequency and importance of words.
CountVectorizer: Used to convert text data into numerical feature vectors based on the frequency of words.

Potential Areas for Improvement or Optimization:

Experiment with different text preprocessing techniques, such as removing stopwords specific to the domain or using more advanced techniques like word embeddings.
Explore other machine learning algorithms or deep learning models for hate speech detection, such as LSTM or BERT.
Perform hyperparameter tuning using techniques like grid search or random search to find the optimal hyperparameters for the models.
Evaluate the models' performance using additional metrics, such as ROC-AUC or precision-recall curves, to gain a more comprehensive understanding of their performance.
Consider using cross-validation techniques to obtain more reliable performance estimates and reduce overfitting.
