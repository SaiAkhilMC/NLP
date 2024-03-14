from transformers import pipeline, set_seed
import warnings
warnings.filterwarnings("ignore")

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

question_answerer = pipeline('question-answering')

question_answerer({
    'question': '?',
    'context': ', " "'})

nlp = pipeline("question-answering")

context = r"""
"""

result = nlp(question="", context=context)

print(f"Answer 1: '{result['answer']}'")

print(result)
result = nlp(question="", context=context)

print(f"Answer 2: '{result['answer']}'")

translator_ger = pipeline("translation_en_to_de")
print("German: ",translator_ger("", max_length=40)[0]['translation_text'])

translator_fr = pipeline('translation_en_to_fr')
print("French: ",translator_fr("",  max_length=40)[0]['translation_text'])

