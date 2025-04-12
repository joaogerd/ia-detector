from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import torch

# Carrega o CSV como Dataset HuggingFace
df = pd.read_csv("../dataset/ia_vs_human.csv")
dataset = Dataset.from_pandas(df)

# Divisão treino/validação
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Modelo e tokenizer
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenização
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Modelo de classificação
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Argumentos de treino
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# Métricas de avaliação
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).mean()
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Treinamento
trainer.train()

# Salvando modelo e tokenizer
model.save_pretrained("../model")
tokenizer.save_pretrained("../model")

