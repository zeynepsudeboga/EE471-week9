from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
print(ner("I am Nate, a research assistant at Izmir Institute of Technology, and currently living "
          "and working in the beautiful city İzmir in Türkiye."))
