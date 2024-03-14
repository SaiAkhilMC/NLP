import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('indiannamesgenders.csv')
df.head()

df=df.dropna()

X = df['name']
y = df['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectorized, y_train)

predictions = svm_model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Accuracy:', accuracy)
print('Classification Report:\n', report)

input_name = input("Enter a name for gender prediction: ")

input_name = input_name.lower()  

input_name_vectorized = vectorizer.transform([input_name])

predicted_gender = svm_model.predict(input_name_vectorized)[0]

print(f'The predicted gender for the name "{input_name}" is: {predicted_gender}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('indiannamesgenders.csv')

df = df.dropna(subset=['name', 'gender'])

X = df['name']
y = df['gender']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max(len(seq) for seq in X_train_sequences)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

embedding_dim = 50
lstm_units = 100

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=lstm_units))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.1)

y_pred_proba = model.predict(X_test_padded)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)

input_name = input("Enter a name for gender prediction: ")

input_name_sequence = tokenizer.texts_to_sequences([input_name])
input_name_padded = pad_sequences(input_name_sequence, maxlen=max_sequence_length)

predicted_proba = model.predict(input_name_padded)
predicted_label = (predicted_proba > 0.5).astype(int)[0][0]

predicted_gender = label_encoder.inverse_transform([predicted_label])[0]

print(f'The predicted gender for "{input_name}" is: {predicted_gender}')



