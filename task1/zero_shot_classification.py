from transformers import pipeline

classifier = pipeline("zero-shot-classification")
print(classifier(
    "Berkshire keeps their cash reserves at an extremely high level.",
    candidate_labels=["finance", "politics", "education"],
))
