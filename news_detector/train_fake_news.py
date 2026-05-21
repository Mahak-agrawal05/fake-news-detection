#python for ml training 

import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real])

# Use only news text
texts = data["title"] + " " + data["text"]

# Labels
labels = data["label"]

# Clean text
cleaned_texts = []

for text in texts:
    
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    cleaned_texts.append(text)

# Convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(cleaned_texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)

# Create model
model = PassiveAggressiveClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# User input
news = input("Enter news: ")

# Clean input
news = news.lower()

news = re.sub(r'[^a-zA-Z\s]', '', news)

# Convert to vector
news_vector = vectorizer.transform([news])

# Predict
result = model.predict_proba(news_vector)

print(result)

# Output
if result[0] == 0:
    print("Fake News")
else:
    print("Real News")

import pickle

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved")