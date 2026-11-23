from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier(
    ["I've been waiting for a modern software practices course my whole life.", "I hate software courses"]
))
