import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def generate_summary(text, num_sentences=5):
    preprocessed_text = preprocess_text(text)

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform([preprocessed_text, text])

    similarity_matrix = cosine_similarity(matrix, matrix)

    sentence_scores = similarity_matrix[0]

    ranked_sentences = sorted(((score, index) for index, score in enumerate(sentence_scores)), reverse=True)

    summary_sentences = [text.split('.')[i] for _, i in ranked_sentences[:num_sentences]]
    summary = '. '.join(summary_sentences)
    return summary

input_text = """
"""

summary = generate_summary(input_text)
print(summary)

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def read_article(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def sentence_similarity(sent1, sent2, stop_words):
    words1 = nltk.word_tokenize(sent1)
    words2 = nltk.word_tokenize(sent2)

    words1 = [word.lower() for word in words1 if word.isalnum() and word.lower() not in stop_words]
    words2 = [word.lower() for word in words2 if word.isalnum() and word.lower() not in stop_words]

    all_words = list(set(words1 + words2))

    vector1 = [1 if word in words1 else 0 for word in all_words]
    vector2 = [1 if word in words2 else 0 for word in all_words]

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)

    return similarity_matrix

def generate_summary(text, num_sentences=5):
    stop_words = set(stopwords.words("english"))

    sentences = read_article(text)

    similarity_matrix = build_similarity_matrix(sentences, stop_words)

    sentence_scores = np.array(similarity_matrix.sum(axis=1))

    ranked_sentences = np.argsort(-sentence_scores)

    summary_sentences = [sentences[i] for i in ranked_sentences[:num_sentences]]
    summary = ' '.join(summary_sentences)

    return summary

input_text = """

"""

summary = generate_summary(input_text)
print(summary)

