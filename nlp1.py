import pandas as pd
import numpy as np
import seaborn as sns

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

from sklearn import metrics                            
from sklearn.metrics import confusion_matrix          
from sklearn.metrics import accuracy_score            
from sklearn.metrics import precision_score           
from sklearn.metrics import recall_score               
from sklearn.metrics import f1_score                   
from sklearn.metrics import classification_report

df = pd.read_csv('spam.csv', delimiter=',', encoding='latin-1')
df.head(8)

df.shape

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df

df.info()

df.describe()

sns.countplot(df.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages');

df['v1'] = df['v1'].map( {'spam': 1, 'ham': 0} )
df.head()

X = df['v2'].values
y = df['v1'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

X_train.shape

X_test.shape

t = Tokenizer()
t.fit_on_texts(X_train)
encoded_train = t.texts_to_sequences(X_train)
encoded_test = t.texts_to_sequences(X_test)
print(encoded_train[0:4])

max_length=8
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
padded_train.shape

vocab_size = len(t.word_index) + 1
model = Sequential()

model.add(Embedding(vocab_size, 24, input_length=max_length))
model.add(SimpleRNN(24, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=padded_train, y=y_train, epochs=50,
         validation_data=(padded_test, y_test), verbose=1)

pred = (model.predict(padded_test) > 0.5).astype("int32")
pred

cm=confusion_matrix(y_test, pred,labels=[0, 1])
df_cm = pd.DataFrame(cm, index = [i for i in ["Actual Ham","Actual Spam"]],
                  columns = [i for i in ["Predicted Ham","Predicted Spam"]])
plt.figure(figsize = (7,5))
plt.title('Confusion Matrix')
sns.heatmap(df_cm, annot=True ,fmt='g');

print('\n Classification Report : \n',metrics.classification_report(y_test, pred))
a = accuracy_score(y_test, pred)
p = precision_score(y_test, pred)
r = recall_score(y_test, pred)
f = f1_score(y_test, pred)

print("Accuracy   : ",round(a,2))
print("Precision  : ",round(p,2))
print("Recall     : ",round(r,2))
print("F1 score   : ",round(f,2))

sms = ["You've Won! Winning an unexpected prize sounds great"]
sms_proc = t.texts_to_sequences(sms)
sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
pred = (model.predict(sms_proc)>0.5).astype("int32").item()
print(pred)
