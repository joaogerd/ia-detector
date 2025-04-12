from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np

# Carregar modelo/tokenizer e dataset
tokenizer = AutoTokenizer.from_pretrained("../model")
model = AutoModelForSequenceClassification.from_pretrained("../model")
df = pd.read_csv("../dataset/ia_vs_human.csv")
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42)["test"]

# Tokenizar
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Avaliação com Trainer
trainer = Trainer(model=model)
preds = trainer.predict(tokenized_dataset)

# Cálculo de acurácia
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids
acc = (y_pred == y_true).mean()
print(f"Acurácia na base de validação: {acc:.4f}")

