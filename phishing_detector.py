from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

emails = [
    "Click here to win free money",
    "Your account has been suspended",
    "Verify your bank details now",
    "Meeting scheduled for tomorrow",
    "Project report attached",
    "Let's have lunch today"
]

labels = [
    "Phishing",
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Safe"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

test_email = ["Click this link and verify account"]
result = model.predict(vectorizer.transform(test_email))

print("Prediction:", result[0])
