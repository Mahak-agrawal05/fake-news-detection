#removing extra words from text

import re
from nltk.corpus import stopwords 

text = "India won the match!!!"

text = text.lower()

text = re.sub(r'[^a-zA-Z\s]', '', text)

words = text.split()

stop_words = stopwords.words('english')
filtered_words = []

for word in words:
    if word not in stop_words:
        filtered_words.append(word)

print(filtered_words)

# import re

# text = "India won the match!!!"

# text = text.lower()

# text = re.sub(r'[^a-zA-Z\s]', '', text)

# words = text.split()

# print(words)