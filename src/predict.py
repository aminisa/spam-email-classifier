import pickle

def classify_email(email):
    with open('models/spam_classifier.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    
    email_vectorized = vectorizer.transform([email])
    
    prediction = model.predict(email_vectorized)[0]
    return "Spam" if prediction == 1 else "Ham"

if __name__ == "__main__":
    email = input("Enter an email to classify: ")
    result = classify_email(email)
    print(f"The email is: {result}")
