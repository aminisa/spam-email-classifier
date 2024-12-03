from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def train_naive_bayes():
    X_tfidf, y = preprocess_data('data/spam_ham_dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    nb_classifier = MultinomialNB()

    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return nb_classifier
