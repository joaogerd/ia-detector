import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar modelo e tokenizer
model_path = "../model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Função de predição
def prever_ia(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        classe = torch.argmax(probs, dim=1).item()
        score = probs[0][1].item()
    return {"Humano": float(1 - score), "IA": float(score)}

# Interface
iface = gr.Interface(
    fn=prever_ia,
    inputs=gr.Textbox(lines=6, placeholder="Digite seu texto aqui..."),
    outputs=gr.Label(num_top_classes=2),
    title="Detector de Texto IA vs Humano",
    description="Este classificador foi treinado para identificar se um texto foi escrito por um humano ou por uma inteligência artificial."
)

if __name__ == "__main__":
    iface.launch()
