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
    
    print("Model training complete and saved.")

if __name__ == "__main__":
    train_model()
