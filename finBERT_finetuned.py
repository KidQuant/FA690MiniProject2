import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import TrainerCallback

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-merged dataset of news and price data
price_df = pd.read_csv('data/price.csv')
news_df = pd.read_csv('data/news.csv')

price_df['daily_return'] = price_df.groupby('ticker')['close'].transform(lambda x: x.pct_change())
price_df = price_df.dropna()

# Convert dates to datetime
news_df['publication_datetime'] = pd.to_datetime(news_df['publication_datetime'])
price_df['Date'] = pd.to_datetime(price_df['Date'])

price_df = price_df.sort_values(['ticker', "Date"])
price_df['daily_return'] = price_df.groupby('ticker')['close'].pct_change()

news_df = news_df.rename(columns={'tickers': 'ticker'})

merged_df = pd.merge_asof(
    news_df.sort_values('publication_datetime'),
    price_df.sort_values('Date'),
    by='ticker',
    left_on='publication_datetime',
    right_on='Date',
    direction='backward'
)
merged_df = merged_df.dropna(subset=['daily_return'])

# Keep the date column for splitting
merged_df = merged_df[["title", "body", "daily_return", "Date"]].dropna()
merged_df["text"] = merged_df["title"] + " " + merged_df["body"]
merged_df["label"] = (merged_df["daily_return"] > 0).astype(int)
merged_df["label"] = merged_df["label"].astype(float)

# Split data based on dates
train_mask = merged_df['Date'] < '2019-01-01'
val_mask = (merged_df['Date'] >= '2019-01-01') & (merged_df['Date'] < '2020-01-01')
test_mask = merged_df['Date'] >= '2020-01-01'

# Create clean dataframes with only the columns we need
train_df = pd.DataFrame({
    'text': merged_df.loc[train_mask, 'text'].values,
    'label': merged_df.loc[train_mask, 'label'].values
})
val_df = pd.DataFrame({
    'text': merged_df.loc[val_mask, 'text'].values,
    'label': merged_df.loc[val_mask, 'label'].values
})
test_df = pd.DataFrame({
    'text': merged_df.loc[test_mask, 'text'].values,
    'label': merged_df.loc[test_mask, 'label'].values
})

# Ensure all values are numeric
train_df['label'] = train_df['label'].astype(float)
val_df['label'] = val_df['label'].astype(float)
test_df['label'] = test_df['label'].astype(float)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Create datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

class FinBERTWithClassifier(nn.Module):
    def __init__(self):
        super(FinBERTWithClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("yiyanghkust/finbert-tone")
        
        # Freeze BERT layers initially
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Enhanced classifier network with batch normalization and more layers
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Add residual connection
        logits = self.classifier(pooled_output).squeeze(-1)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  
    logging_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Reduced batch size for better generalization
    per_device_eval_batch_size=16,
    num_train_epochs=15,  # Increased epochs
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,  
    metric_for_best_model="eval_loss",
    fp16=True,
    warmup_ratio=0.1,  # Add warmup
    gradient_accumulation_steps=4,  # Accumulate gradients
)

model = FinBERTWithClassifier()

def compute_metrics(pred):
    # Handle the prediction format correctly
    if isinstance(pred, tuple):
        if len(pred) == 2:
            logits, labels = pred
        else:
            logits = pred[0]
            labels = pred[1]
    else:
        logits = pred.predictions
        labels = pred.label_ids
    
    # Ensure we have the right shape
    if len(logits.shape) > 1:
        logits = logits.squeeze()
    
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Ensure we're working with float arrays
    logits = logits.astype(np.float32)
    labels = labels.astype(np.float32)
    
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    
    # Calculate additional metrics
    mse = mean_squared_error(labels, probs)
    r2 = r2_score(labels, probs)
    
    # Add regularization to R2 score
    adjusted_r2 = 1 - (1 - r2) * (len(labels) - 1) / (len(labels) - probs.shape[0] - 1)
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "mse": mse,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "f1": f1_score(labels, preds)
    }
    return metrics

# Initialize dictionaries to store metrics
train_metrics = defaultdict(list)
val_metrics = defaultdict(list)

class MetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                train_metrics["loss"].append(logs["loss"])
            if "eval_loss" in logs:
                val_metrics["loss"].append(logs["eval_loss"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[MetricsCallback()]
)

trainer.train()

# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
if train_metrics["loss"]:  # Check if we have training loss data
    plt.plot(train_metrics["loss"], label="Training Loss")
if val_metrics["loss"]:  # Check if we have validation loss data
    plt.plot(val_metrics["loss"], label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig("loss_curves.png")
plt.show()

# After training, evaluate on test set
test_results = trainer.evaluate(test_dataset)
print("\nTest Set Results:")
print(test_results)

# Get predictions for the test set
test_preds = trainer.predict(test_dataset)
test_probs = 1 / (1 + np.exp(-test_preds.predictions))
test_labels = test_preds.label_ids

test_metrics = {
    "accuracy": accuracy_score(test_labels, (test_probs > 0.5).astype(int)),
    "mse": mean_squared_error(test_labels, test_probs),
    "r2": r2_score(test_labels, test_probs),
    "f1": f1_score(test_labels, (test_probs > 0.5).astype(int))
}
print("\nTest Set Metrics:")
print(test_metrics)