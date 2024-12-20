from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    X = df['text']
    y = df['label_num']
    
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer

'''
Purpose: Preprocesses the text data for the model.
'''

'''
Role: Converts email text into numerical vectors using CountVectorizer. Splits the dataset into training and testing sets while preserving class distribution.
Outputs the training and testing data along with the vectorizer for consistent data transformation.
'''

'''
Dependencies: Requires a DataFrame (cleaned by data_loader.py).
'''