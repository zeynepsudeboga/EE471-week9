from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

reloaded = AutoModelForSequenceClassification.from_pretrained("saved_model")
print("Saved files:", sorted(__import__("os").listdir("saved_model")))
print("Reloaded labels:", reloaded.config.id2label)
