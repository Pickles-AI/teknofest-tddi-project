from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class NERService:
    def __init__(self):
        self.model_name = "moarslan/bert-base-turkish-cased-ner"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="max")

    def analyze(self, text):
        print("NER çalışıyor..")
        ner_results = self.ner_pipeline(text)
        return [item for item in ner_results if item['entity_group'] != 'O']