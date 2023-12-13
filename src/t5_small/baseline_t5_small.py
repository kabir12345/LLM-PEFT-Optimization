import os
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from datasets import load_dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt')

# Define the path to the cache directory
cache_dir ="/Users/kabir/Downloads/LLM-PEFT-Optimization/cache"

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Load and preprocess the CNN/Daily Mail dataset
def load_and_preprocess_data(tokenizer):
    cache_file = os.path.join(cache_dir, "cnn_dailymail_tokenized_t5")

    if os.path.exists(cache_file):
        logging.info("Loading tokenized dataset from cache")
        return DatasetDict.load_from_disk(cache_file)

    logging.info("Processing and tokenizing dataset (this may take a while)")
    dataset = load_dataset("cnn_dailymail", '3.0.0')

    def tokenize_function(examples):
        inputs = tokenizer("summarize: " + examples["article"], padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(examples["highlights"], padding="max_length", truncation=True, max_length=128)
        return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": targets.input_ids}

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.save_to_disk(cache_file)
    return tokenized_dataset

# Initialize T5-small model
def initialize_model():
    return T5ForConditionalGeneration.from_pretrained("t5-small")

# Compute metrics
def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {key: scorer.score(ref, pred)[key].fmeasure for key, pred, ref in zip(['rouge1', 'rouge2', 'rougeL'], predictions, references)}
    bleu_scores = [sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred), smoothing_function=SmoothingFunction().method1) for pred, ref in zip(predictions, references)]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return rouge_scores, avg_bleu_score


# Training function
def train_and_evaluate(model, train_loader, val_dataset, tokenizer, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch['input_ids'], batch['labels']
            attention_mask = batch['attention_mask']
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            trimmed_logits = logits[:, :labels.size(1), :]
            loss = torch.nn.CrossEntropyLoss()(trimmed_logits.view(-1, trimmed_logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        predictions, references = [], []
        for batch in val_dataset:
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            with torch.no_grad():
                outputs = model.generate(inputs, attention_mask=attention_mask)
            decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
            decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

        rouge_scores, avg_bleu_score = compute_metrics(predictions, references)
        logging.info(f"Epoch {epoch}: ROUGE: {rouge_scores}, BLEU: {avg_bleu_score}")

# Main function
def main():
    tokenized_dataset = load_and_preprocess_data(tokenizer)
    model = initialize_model()
    train_loader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_and_evaluate(model, train_loader, tokenized_dataset['validation'], tokenizer, optimizer)

if __name__ == "__main__":
    main()
