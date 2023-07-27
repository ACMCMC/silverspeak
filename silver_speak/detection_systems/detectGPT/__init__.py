from ..detectionSystem import DetectionSystem
from .detectGPT import GPT2PPLV2 as GPT2PPL

class DetectGPT(DetectionSystem):
    def __init__(self):
        super().__init__()
        self.model = GPT2PPL()
    
    def detect(self, text):
        outputs = self.model(text, 100, "v1.1")
        prob = float(outputs[0]['prob'][:-1]) / 100.0
        return {
            'predicted': 'human' if outputs[0]['label'] == 1 and prob > 0.5 else 'generated',
            'probability': prob
            }
