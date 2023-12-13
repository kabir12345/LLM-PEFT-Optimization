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
from peft import LoraConfig, IA3Config, get_peft_model 
from torch.optim import AdamW  # Import from PyTorch


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt')

# Define the path to the cache directory
cache_dir = "/Users/kabir/Downloads/LLM-PEFT-Optimization/cache/"  # Update this path


tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
def collate_fn(batch):
    input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(item['labels']) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

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

def initialize_lora_model():
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,  # LoRA alpha
        target_modules=["attn.c_attn", "attn.c_proj"],  # Target modules for LoRA
        fan_in_fan_out=True  # Set to True for GPT-2's Conv1D layers
    )
    lora_model = get_peft_model(base_model, lora_config)
    return lora_model


# Initialize GPT-2 model with IA3
def initialize_ia3_model():
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    ia3_config = IA3Config(
        target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],  # Include both attention and feedforward modules
        feedforward_modules=["mlp.c_fc", "mlp.c_proj"]  # Feedforward modules
    )
    ia3_model = get_peft_model(base_model, ia3_config)
    return ia3_model


def get_optimizer(model):
    base_params = [p for n, p in model.named_parameters() if not n.endswith("_peft")]
    peft_params = [p for n, p in model.named_parameters() if n.endswith("_peft")]
    optimizer = AdamW([
        {'params': base_params},
        {'params': peft_params, 'lr': 1e-4}  # Higher learning rate for PEFT parameters
    ], lr=5e-5)
    return optimizer

def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {key: scorer.score(ref, pred)[key].fmeasure for key, pred, ref in zip(['rouge1', 'rouge2', 'rougeL'], predictions, references)}
    bleu_scores = [sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred), smoothing_function=SmoothingFunction().method1) for pred, ref in zip(predictions, references)]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return rouge_scores, avg_bleu_score

def train_and_evaluate(mod_name,model, train_loader, val_loader, tokenizer, optimizer, epochs=2):
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
    img_path='/Users/kabir/Downloads/LLM-PEFT-Optimization/results/gpt2/optimised_gpt2_'+mod_name+'.png'
    plt.savefig(img_path)  # Update the path to your directory
    plt.show()

def main():
    TRAIN_DATA_LIMIT = 4
    VAL_DATA_LIMIT = 4
    tokenized_dataset = load_and_preprocess_data(tokenizer, train_limit=TRAIN_DATA_LIMIT, val_limit=VAL_DATA_LIMIT)
    train_loader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_dataset['validation'], batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Train and evaluate LoRA model
    # lora_model = initialize_lora_model()
    # lora_optimizer = get_optimizer(lora_model)
    # lora_model_name="lora"
    # train_and_evaluate(lora_model_name,lora_model, train_loader, val_loader, tokenizer, lora_optimizer)

    # Train and evaluate IA3 model
    ia3_model = initialize_ia3_model()
    ia3_optimizer = get_optimizer(ia3_model)
    ia3_model_name="ia3"
    train_and_evaluate(ia3_model_name,ia3_model, train_loader, val_loader, tokenizer, ia3_optimizer)



if __name__ == "__main__":
    main()
