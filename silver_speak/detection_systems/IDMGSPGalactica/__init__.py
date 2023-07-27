from ..detectionSystem import DetectionSystem
from transformers import pipeline, AutoTokenizer, OPTForSequenceClassification

class IDMGSPGalactica(DetectionSystem):
    def __init__(self):
        super().__init__()
        model = OPTForSequenceClassification.from_pretrained("tum-nlp/IDMGSP-Galactica-TRAIN")
        tokenizer = AutoTokenizer.from_pretrained("tum-nlp/IDMGSP-Galactica-TRAIN")
        self.reader = pipeline("text-classification", model=model, tokenizer = tokenizer, max_length=2048)
    
    def detect(self, text):
        outputs = self.reader(text)
        return {
            'predicted': 'human' if outputs[0]['label'] == 'real' else 'generated',
            'probability': outputs[0]['score']
            }