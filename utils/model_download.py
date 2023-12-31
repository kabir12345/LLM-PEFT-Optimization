import logging
from transformers import GPT2Model, DistilBertModel, T5Model
logging.basicConfig(level=logging.INFO)
custom_cache_dir = "/Users/kabir/Downloads/LLM-PEFT-Optimization/data/models"
gpt2_model = GPT2Model.from_pretrained("gpt2", cache_dir=custom_cache_dir)
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", cache_dir=custom_cache_dir)
t5_model = T5Model.from_pretrained("t5-small", cache_dir=custom_cache_dir)
