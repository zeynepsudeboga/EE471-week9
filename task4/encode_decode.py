from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentence = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(sentence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens :", tokens)
print("IDs    :", ids)
print("Decoded:", tokenizer.decode(ids))
