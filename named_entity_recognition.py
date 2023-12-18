# Implements Named Entity Recognition functionalities
# transfoermer source:https://huggingface.co/dslim/bert-base-NER

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Implement NER pipeline
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer)

def extract_name(text):
    # Extract the user's name
    results = ner_pipeline(text)
    name = ""
    for result in results:
        if 'PER' in result["entity"]:
            name = result["word"]
    return name


