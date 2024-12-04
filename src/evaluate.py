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

'''
Purpose: Evaluates the performance of the trained model.
'''

'''
Role: Reloads the dataset using data_loader.py.
Preprocesses the data using preprocess.py.
Loads the saved model from spam_classifier.pkl and evaluates it on the test set.
Outputs metrics such as precision, recall, F1-score, and accuracy to quantify the model's performance.
'''

'''
Dependencies: Uses spam_classifier.pkl from train.py.
Relies on data_loader.py and preprocess.py for input data preparation.
'''