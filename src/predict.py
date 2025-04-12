from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys

# Carrega modelo e tokenizer
model_path = "../model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Função de predição
def prever(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        score = probs[0][1].item()
    return pred, score

# Execução direta com input de texto
if __name__ == "__main__":
    texto = input("Digite um texto para classificar como IA (1) ou Humano (0):\n")
    pred, score = prever(texto)
    print(f"\nResultado: {'IA' if pred == 1 else 'Humano'} (score IA: {score:.4f})")


