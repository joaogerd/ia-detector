# 🧠 IA Detector (Português)

Este projeto implementa um classificador de textos para identificar se um conteúdo foi escrito por um ser humano ou gerado por inteligência artificial, utilizando o modelo BERTimbau.

---

## 📂 Estrutura do Projeto

```
ia_detector/
├── dataset/
│   └── ia_vs_human.csv         # Dataset rotulado: textos humanos (0) e IA (1)
├── model/                      # Modelo treinado será salvo aqui
├── notebooks/
│   └── analyze.ipynb           # Visualizações e análise do modelo
├── src/
│   ├── train.py                # Script de fine-tuning com BERTimbau
│   ├── evaluate.py             # Avaliação do modelo
│   ├── predict.py              # Inferência com novos textos
│   └── app.py                  # Interface web com Gradio
├── requirements.txt           # Dependências
├── environment.yaml           # Ambiente Conda
├── Dockerfile                 # Container para execução
├── LICENSE                    # Licença do projeto
├── README.md                  # Instruções de uso
├── .gitignore                 # Ignora arquivos sensíveis e gerados automaticamente
```

---

## 🚀 Como Usar

### 1. Criar ambiente (Conda recomendado)
```bash
conda env create -f environment.yaml
conda activate ia-detector
```

Ou usar:
```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo
```bash
cd src
python train.py
```

### 3. Avaliar desempenho
```bash
python evaluate.py
```

### 4. Predizer com texto manual
```bash
python predict.py
```

### 5. Iniciar a interface Web (Gradio)
```bash
python app.py
```

---

## 📈 Visualização de Resultados
Abra o notebook:
```bash
jupyter notebook notebooks/analyze.ipynb
```
E veja:
- Matriz de confusão
- Curva ROC
- Distribuição das probabilidades
- Relatório de classificação

---

## 🐳 Docker

### Build e run local
```bash
docker build -t ia-detector .
docker run -p 7860:7860 ia-detector
```

Interface disponível em `http://localhost:7860`

---

## 📊 Dataset
- Textos humanos extraídos de obras literárias (label: `0`)
- Textos IA gerados automaticamente (label: `1`)

---

## 🤖 Modelo Utilizado
- [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

---

## 📋 Licença
MIT ou LGPL v3 (dependendo da distribuição escolhida).

---

Feito por João Gerd Zell de Mattos com ❤️, Python e café ☕
