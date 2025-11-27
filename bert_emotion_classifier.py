from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

class BERTEmotionClassifier:
    def __init__(self, model_name="indobenchmark/indobert-base-p2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=4
        ).to(self.device)
        self.emotion_labels = ["happy", "sad", "angry", "calm"]
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict_emotion(self, text):
        result = self.classifier(text, truncation=True, max_length=512)
        emotion_idx = int(result[0]["label"].split("_")[-1])
        return self.emotion_labels[emotion_idx]

# Singleton instance
_bert_classifier = None

def get_bert_classifier():
    global _bert_classifier
    if _bert_classifier is None:
        _bert_classifier = BERTEmotionClassifier()
    return _bert_classifier
