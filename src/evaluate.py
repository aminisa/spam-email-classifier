import pickle
from sklearn.metrics import classification_report
from data_loader import load_data
from preprocess import preprocess_data

def evaluate_model():
    file_path = 'data/spam_ham_dataset.csv'
    df = load_data(file_path)
    
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)
    
    with open('models/spam_classifier.pkl', 'rb') as f:
        model, _ = pickle.load(f)
    
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

if __name__ == "__main__":
    evaluate_model()
