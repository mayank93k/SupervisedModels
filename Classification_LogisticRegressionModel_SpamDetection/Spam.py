from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample Data
emails = [
    "Win a free iPhone now!",
    "Meeting at 10 AM tomorrow",
    "Congratulations, you won $1000!",
    "Can we reschedule the meeting?"
]
labels = [1, 0, 1, 0]  # Spam=1, Not Spam=0

# Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Initialize and train the model
model = LogisticRegression()
model.fit(X, labels)

# Predict for a new email
new_email = ["Claim your free prize today!"]
X_new = vectorizer.transform(new_email)
prediction = model.predict(X_new)
print("Spam" if prediction[0] == 1 else "Not Spam")
