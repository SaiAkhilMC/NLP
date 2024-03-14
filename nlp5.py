from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

sentences = [
    "Machine learning is fascinating",
    "Deep learning is important for AI",
    "Machine Learning, Deep Learning are crucial for NLP"
]

tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, sg=0, min_count=1, workers=4)

word = "learning"
word_vector = model.wv[word]

print("Word:", word)
print("Word2Vec representation using CBOW:", word_vector)

