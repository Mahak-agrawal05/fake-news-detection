from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data
sentences = [
    "India won the cricket match",
    "Virat Kohli scored century",
    "Aliens attacked earth",
    "Spaceship landed on mars"
]

# Labels
# 0 = Sports
# 1 = Space

labels = [0, 0, 1, 1]

# Convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sentences)

# Create model
model = LogisticRegression()

# Train model
model.fit(X, labels)

# Test sentence
test = ["India won the game"]

# Convert test sentence
test_vector = vectorizer.transform(test)

# Predict
prediction = model.predict(test_vector)

print(prediction)