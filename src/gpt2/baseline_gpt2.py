import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import logging
import matplotlib.pyplot as plt


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt')

# Define the path to the cache directory
cache_dir = "/Users/kabir/Downloads/LLM-PEFT-Optimization/cache"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

def collate_fn(batch):
    input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(item['labels']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Load and preprocess the CNN/Daily Mail dataset
def load_and_preprocess_data(tokenizer, train_limit=None, val_limit=None):
    cache_file = os.path.join(cache_dir, "cnn_dailymail_tokenized")
    if os.path.exists(cache_file):
        logging.info("Loading tokenized dataset from cache")
        dataset = DatasetDict.load_from_disk(cache_file)
    else:
        logging.info("Processing and tokenizing dataset (this may take a while)")
        dataset = load_dataset("cnn_dailymail", '3.0.0')
        def tokenize_function(examples):
            inputs = tokenizer(examples["article"], padding="max_length", truncation=True, max_length=512)
            targets = tokenizer(examples["highlights"], padding="max_length", truncation=True, max_length=512)
            return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": targets.input_ids}
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.save_to_disk(cache_file)

    if train_limit:
        dataset['train'] = Subset(dataset['train'], range(train_limit))
    if val_limit:
        dataset['validation'] = Subset(dataset['validation'], range(val_limit))

    return dataset


# Initialize GPT-2 model
def initialize_model():
    return GPT2LMHeadModel.from_pretrained("gpt2")

# Compute metrics
def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {key: scorer.score(ref, pred)[key].fmeasure for key, pred, ref in zip(['rouge1', 'rouge2', 'rougeL'], predictions, references)}
    bleu_scores = [sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred), smoothing_function=SmoothingFunction().method1) for pred, ref in zip(predictions, references)]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return rouge_scores, avg_bleu_score

# Training function
def train_and_evaluate(model, train_loader, val_loader, tokenizer, optimizer, epochs=10):
    # Lists to keep track of metrics for plotting
    epoch_losses = []
    epoch_rouge1_scores = []
    epoch_rouge2_scores = []
    epoch_rougeL_scores = []
    epoch_bleu_scores = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
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
            batch_losses.append(loss.item())

        # Calculate average loss for the epoch
        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)

        # Evaluation
        model.eval()
        predictions, references = [], []
        for batch in val_loader:
            inputs, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            with torch.no_grad():
                generated_outputs = model.generate(inputs, attention_mask=attention_mask, max_length=512 + 50)
                decoded_outputs = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_outputs]
                decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            predictions.extend(decoded_outputs)
            references.extend(decoded_labels)

        rouge_scores, avg_bleu_score = compute_metrics(predictions, references)
        epoch_rouge1_scores.append(rouge_scores['rouge1'])
        epoch_rouge2_scores.append(rouge_scores['rouge2'])
        epoch_rougeL_scores.append(rouge_scores['rougeL'])
        epoch_bleu_scores.append(avg_bleu_score)

        logging.info(f"Epoch {epoch}: Loss: {avg_loss}, ROUGE: {rouge_scores}, BLEU: {avg_bleu_score}")

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epoch_rouge1_scores, label='ROUGE-1')
    plt.plot(epoch_rouge2_scores, label='ROUGE-2')
    plt.plot(epoch_rougeL_scores, label='ROUGE-L')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE Score')
    plt.title('ROUGE Scores vs Epoch')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epoch_bleu_scores, label='BLEU')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score vs Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/kabir/Downloads/LLM-PEFT-Optimization/results/gpt2/baseline_gpt2.png')  # Update the path to your directory
    plt.show()
# Main function
def main():
    TRAIN_DATA_LIMIT = 20  # Adjust this to limit the amount of training data
    VAL_DATA_LIMIT = 20   # Adjust this to limit the amount of validation data
    tokenized_dataset = load_and_preprocess_data(tokenizer, train_limit=TRAIN_DATA_LIMIT, val_limit=VAL_DATA_LIMIT)
    model = initialize_model()
    train_loader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_dataset['validation'], batch_size=4, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_and_evaluate(model, train_loader, val_loader, tokenizer, optimizer)

if __name__ == "__main__":
    main()
