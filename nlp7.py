import nltk
import numpy as np
import random
import string

nltk.download('punkt')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

corpus = [
    
]

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(token) for token in text if token not in string.punctuation]
    return ' '.join(text)

corpus = [preprocess_text(sentence) for sentence in corpus]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def generate_response(user_input):
    user_input = preprocess_text(user_input)
    tfidf_user_input = vectorizer.transform([user_input])
    similarity = cosine_similarity(tfidf_user_input, tfidf_matrix)
    max_similarity_index = np.argmax(similarity)
    return corpus[max_similarity_index]

print("Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Chatbot: Goodbye! Have a great day!")
        break
    else:
        response = generate_response(user_input)
        print("Chatbot:", response)

