#flask backend

from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load saved model
model = pickle.load(open("model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    # Clean text
    news = news.lower()

    news = re.sub(r'[^a-zA-Z\s]', '', news)

    # Convert to vector
    news_vector = vectorizer.transform([news])

    # Predict
    result = model.predict(news_vector)

    if result[0] == 0:
        prediction = "Fake News"
    else:
        prediction = "Real News"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)