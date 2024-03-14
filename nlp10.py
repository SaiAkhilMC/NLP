doc_1 = ""
doc_2 = ""
doc_3 = ""
doc_4 = ""
doc_5 = ""
corpus = [doc_1, doc_2, doc_3, doc_4, doc_5]

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

clean_corpus = [clean(doc).split() for doc in corpus]

from gensim import corpora
dictionary = corpora.Dictionary(clean_corpus)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]

from gensim.models import LsiModel
lsa = LsiModel(doc_term_matrix, num_topics=3, id2word = dictionary)
print(lsa.print_topics(num_topics=3, num_words=3))

from gensim.models import LdaModel
lda = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary)
print(lda.print_topics(num_topics=3, num_words=3))




