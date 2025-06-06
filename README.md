# Song Lyrics Classification Project

Overview:

This project classifies song lyrics into genres (Pop, Rap, Miscellaneous) using Natural Language Processing (NLP) and Support Vector Machine (SVM) techniques. The dataset, sourced from Genius via Kaggle, contains 5 million songs, enhanced with language detection models to identify the native language of each entry. The project focuses on three genres due to uneven distribution in the original dataset and achieves an accuracy of 86.11% using an SVM model with an RBF kernel.


Project Workflow:

The project is divided into four main stages:





Data Collection: A 9GB dataset of 5 million songs was downloaded from Kaggle, categorized into genres (Pop, Rock, Country, Rap, R&B, Miscellaneous).



Data Preparation: Selected three genres (Pop, Rap, Miscellaneous) with balanced sampling (~10,000 songs per category). Cleaned lyrics by converting to lowercase, expanding contractions (e.g., "don't" to "do not"), and removing punctuation, numbers, and special characters. Converted lyrics into numerical feature vectors using parameters like min_df and max_df to filter terms. Applied chi-squared test to select the top 20,000 features.



Model Training: Used GridSearchCV with 5-fold cross-validation (StratifiedKFold, n_splits=5, shuffle=True, random_state=42) to find optimal SVM hyperparameters.



Best parameters: C=1, kernel='rbf', class_weight=None.



Model Evaluation: Evaluated using accuracy, precision, recall, and F1-score via a confusion matrix.



Results:





Overall accuracy: 86.11%



Precision: Pop (0.81), Miscellaneous (0.88), Rap (0.90)



Recall: Pop (0.84), Miscellaneous (0.87), Rap (0.83)



F1-score: Pop (0.82), Miscellaneous (0.87), Rap (0.86)

The SVM model with an RBF kernel outperformed a linear kernel, capturing non-linear relationships in the data.

Steps to do:

1. Run split_data.py

2. Run prepare-data.py

![image](https://github.com/user-attachments/assets/d0b1e41e-b789-4ce6-b25c-82ff6b9c7202)


3. Run svm_classification.py

![image](https://github.com/user-attachments/assets/81c62a74-8432-4246-a48d-c88fa93efdd6)
![image](https://github.com/user-attachments/assets/2c2485db-0eb6-4799-9580-63309a9c8623)

sample for each category=10000, k-fold cross validation (5), min_df =100, rbf kernel

4. Retrain the model by typing svm-train -t 0 data/libsvm_train.txt model.txt in terminal and svm-predict data/libsvm_test.txt model.txt predictions.txt in terminal 


Limitations:

1. Dataset size (9GB) poses memory and computational challenges.
2. 17% misclassification rate for Rap lyrics.
3. Limited to three genres due to uneven distribution.
4. High resource requirements for feature extraction and model training.

Acknowledgments:

-Inspired by the GitHub repository johnsonj561/LibSVM-Text-Classification.

-Dataset sourced from Kaggle and Genius ([https://genius.com](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)).
