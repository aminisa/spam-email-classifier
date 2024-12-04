import os
import pickle
from data_loader import load_data
from preprocess import preprocess_data
from sklearn.naive_bayes import MultinomialNB

def train_model():
    file_path = 'data/spam_ham_dataset.csv'
    df = load_data(file_path)
    
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    with open('models/spam_classifier.pkl', 'wb') as f:
        pickle.dump((model, vectorizer), f)
    
    print("Model training complete and saved.\n")

if __name__ == "__main__":
    train_model()

'''
Purpose: Trains the Naive Bayes classifier on the preprocessed data.
'''

'''
Role: Uses data_loader.py to load the dataset. Calls preprocess.py to preprocess the data. Fits the MultinomialNB model on the training set.
Saves the trained model and vectorizer as spam_classifier.pkl for reuse.
'''

'''
Dependencies: Relies on data_loader.py and preprocess.py to prepare the data.
Saves outputs to the models folder for downstream tasks.
'''