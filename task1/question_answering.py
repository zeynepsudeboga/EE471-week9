from transformers import pipeline

context = ("I am Nate, a research assistant at Izmir Institute of Technology, and currently living "
           "and working in the beautiful city İzmir in Türkiye.")
question_answerer = pipeline("question-answering")
print(question_answerer(question="What is my name?", context=context))
print(question_answerer(question="Where do I work?", context=context))
print(question_answerer(question="Which city do I live in?", context=context))
