import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(" " +
          " ")

rows = []
rows.append(["Word", "Position", "Lowercase", "Lemma", "POS", "Alphanumeric", "Stopword"])
for token in doc:
    rows.append([token.text, str(token.i), token.lower_, token.lemma_,
                 token.pos_, str(token.is_alpha), str(token.is_stop)])

columns = zip(*rows)

column_widths = [max(len(item) for item in col) for col in columns]

for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
        for i in range(0, len(row))))

import spacy
NER = spacy.load("en_core_web_sm")
raw_text=""
text1= NER(raw_text)
for word in text1.ents:
    print(word.text,word.label_)

import spacy

nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."

doc = nlp(text)

pos_tags = [(token.text, token.pos_) for token in doc]

print(pos_tags)





