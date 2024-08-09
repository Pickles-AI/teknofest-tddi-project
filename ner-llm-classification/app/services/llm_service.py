from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class LLMService:
    def __init__(self):
        self.model_name = "google/flan-t5-small"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Kullanılan cihaz: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def analyze_sentiment(self, text, entity):
        prompt = """Sen bir duygu analizi asistanısın. Sana verilen metindeki belirtilen varlıklar (entities) hakkındaki duyguları analiz et ve her bir varlık için duygu durumunu (olumlu, olumsuz veya nötr) belirle. Yanıtını JSON formatında ver.

        Örnekler:
        1. Metin: "Netflix'te son dönemde izlediğim dizilerden çok memnunum, ancak Amazon Prime'ın sunduğu seçenekler pek tatmin edici değil."
           Varlıklar: ["netflix", "amazon prime"]
           Duygu Durumları: {"netflix": "Olumlu", "amazon prime": "Olumsuz"}

        2. Metin: "Yeni iPhone gerçekten harika bir cihaz, ancak Samsung'un kamera kalitesi hala bir adım önde."
           Varlıklar: ["iphone", "samsung"]
           Duygu Durumları: {"iphone": "Olumlu", "samsung": "Olumlu"}
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.3,
            top_p=0.9,
            do_sample=False,
            num_beams=3,
        )

        duygu_analizi = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Duygu durumunu analizden çıkar
        duygu = duygu_analizi.split('.')[0].strip().lower()
        if duygu not in ['olumlu', 'olumsuz', 'nötr']:
            duygu = 'nötr'  # Eğer duygu belirlenemezse nötr olarak varsay

        return duygu