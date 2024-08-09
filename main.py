import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn.functional as F

app = FastAPI()

# NER modeli
ner_model_name = "moarslan/bert-base-turkish-cased-ner"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_pipeline = pipeline(
    "ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="average"
)

# ABSA modeli
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
absa_tokenizer = AutoTokenizer.from_pretrained(absa_model_name)
absa_model = AutoModelForSequenceClassification.from_pretrained(absa_model_name)

# Geleneksel Duygu Analizi modeli
sentiment_model_name = "moarslan/bert-base-turkish-sentiment-analysis"
sentiment_model = pipeline(
    "sentiment-analysis", model=sentiment_model_name, tokenizer=sentiment_model_name
)


class TextInput(BaseModel):
    text: str


def get_sentiment(text, aspect):
    inputs = absa_tokenizer(
        f"[CLS] {text} [SEP] {aspect} [SEP]",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0]
    predicted_class = np.argmax(probs)
    sentiment_label = ["olumsuz", "nötr", "olumlu"][predicted_class]
    return sentiment_label, probs.tolist()


def analyze_text(text):
    # NER analizi
    ner_results = ner_pipeline(text)

    # Varlıkları çıkar
    entities = [
        entity["word"]
        for entity in ner_results
        if entity["entity_group"] in ["ORG", "PRODUCT"]
    ]
    entities = list(dict.fromkeys(entities))  # Tekrarları kaldır

    # ABSA ve geleneksel duygu analizi
    results = []
    for entity in entities:
        absa_sentiment, absa_probs = get_sentiment(text, entity)
        traditional_sentiment = sentiment_model(entity)[0]

        results.append(
            {
                "entity": entity,
                "sentiment": absa_sentiment,
            }
        )

    # Sonuçları JSON formatında döndür
    output = {
        "entity_list": entities,
        "results": results,
    }
    return output


@app.post("/predict")
async def predict(input: TextInput):
    return analyze_text(input.text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
