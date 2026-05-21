#TF-IDF VECTORIZATION  words → numbers

from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "India won the match",
    "Aliens attacked earth",
    "India lost the game"
]

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(sentences)

print(vectorizer.get_feature_names_out())

print(vectors.toarray())