"""Week 9 prep: runs every pipeline() example of the Hugging Face LLM course, chapter 1/3.
The first run downloads each model to ~/.cache/huggingface; in class they load from there."""
from transformers import pipeline


def show(title, output):
    print(f"\n===== {title}\n{output}", flush=True)


classifier = pipeline("sentiment-analysis")
show("sentiment-analysis", classifier(["I've been waiting for a HuggingFace course my whole life.",
                                       "I hate this so much!"]))

classifier = pipeline("zero-shot-classification")
show("zero-shot-classification", classifier("This is a course about the Transformers library",
                                            candidate_labels=["education", "politics", "business"]))

# pipeline("text-generation") without a model now defaults to SmolLM3-3B (~6 GB), skipped for disk space.
# The course's own model for this example, SmolLM2-360M, is downloaded in the next block.

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
show("text-generation (SmolLM2-360M)", generator("In this course, we will teach you how to",
                                                  max_length=30, num_return_sequences=2))

unmasker = pipeline("fill-mask")
show("fill-mask", unmasker("This course will teach you all about <mask> models.", top_k=2))

ner = pipeline("ner", aggregation_strategy="simple")   # the course's grouped_entities=True (newer name)
show("ner", ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

question_answerer = pipeline("question-answering")
show("question-answering", question_answerer(question="Where do I work?",
                                             context="My name is Sylvain and I work at Hugging Face in Brooklyn"))

summarizer = pipeline("summarization")
show("summarization", summarizer(
    "America has changed dramatically during recent years. Not only has the number of graduates in "
    "traditional engineering disciplines such as mechanical, civil, electrical, chemical, and aeronautical "
    "engineering declined, but in most of the premier American universities engineering curricula now "
    "concentrate on and encourage largely the study of engineering science."))

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
show("translation", translator("Ce cours est produit par Hugging Face."))

image_classifier = pipeline(task="image-classification", model="google/vit-base-patch16-224")
show("image-classification", image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"))

transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
show("automatic-speech-recognition", transcriber(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))