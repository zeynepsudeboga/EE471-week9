"""app.py – Gradio front end: one tab per task, all using the HuggingFaceTasks class.

Run:  python app.py   then open http://127.0.0.1:7860
"""
import gradio as gr

from hf_tasks import HuggingFaceTasks

tasks = HuggingFaceTasks()

NATE = ("I am Nate, a research assistant at Izmir Institute of Technology, and currently living "
        "and working in the beautiful city İzmir in Türkiye.")
CRISIS = ("The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st "
          "century, often compared to the Great Depression of the 1930s. Triggered by the bursting of the "
          "United States housing bubble, its effects rippled across the globe, leading to the collapse of "
          "major financial institutions and a deep international recession. The crisis began with the "
          "subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership "
          "led banks to issue high-risk loans to borrowers with poor credit.")


# Small adapters: turn each method's result into what a Gradio component shows.
def sentiment(text):
    r = tasks.sentiment(text)
    return {r["label"]: r["score"]}


def zero_shot(text, labels):
    return tasks.zero_shot(text, [l.strip() for l in labels.split(",") if l.strip()])


def generate(prompt, max_new_tokens):
    return "\n\n---\n\n".join(tasks.generate(prompt, int(max_new_tokens)))


def fill_mask(text):
    return {r["sequence"]: r["score"] for r in tasks.fill_mask(text)}


def ner(text):
    return {"text": text, "entities": tasks.ner(text)}       # HighlightedText reads entity_group/start/end


def answer(question, context):
    r = tasks.answer(question, context)
    return f"{r['answer']}   (score {r['score']:.2f})"


def classify_image(image):
    return {r["label"]: r["score"] for r in tasks.classify_image(image)}


with gr.Blocks(title="EE471 – Hugging Face tasks") as demo:
    gr.Markdown("# EE471 Week 9 – Hugging Face tasks\nEvery tab calls one method of the `HuggingFaceTasks` class.")

    with gr.Tab("Sentiment analysis"):
        inp = gr.Textbox(label="Text", value="I've been waiting for a modern software practices course my whole life.")
        out = gr.Label(label="Sentiment")
        gr.Button("Analyze").click(sentiment, inp, out)

    with gr.Tab("Zero-shot classification"):
        inp = gr.Textbox(label="Text", value="Berkshire keeps their cash reserves at an extremely high level.")
        labels = gr.Textbox(label="Candidate labels (comma separated)", value="finance, politics, education")
        out = gr.Label(label="Scores")
        gr.Button("Classify").click(zero_shot, [inp, labels], out)

    with gr.Tab("Text generation"):
        inp = gr.Textbox(label="Start of the sentence",
                         value="If I continue to complete all in-class exercises in this modern software practices course,")
        length = gr.Slider(10, 80, value=35, step=1, label="Max new tokens")
        out = gr.Textbox(label="Two continuations", lines=8)
        gr.Button("Generate").click(generate, [inp, length], out)

    with gr.Tab("Mask filling"):
        inp = gr.Textbox(label="Text with <mask> or [MASK]",
                         value="To understand generative AI, one must study <mask> well.")
        out = gr.Label(label="Top predictions")
        gr.Button("Fill").click(fill_mask, inp, out)

    with gr.Tab("Named entity recognition"):
        inp = gr.Textbox(label="Text", value=NATE)
        out = gr.HighlightedText(label="Entities")
        gr.Button("Find entities").click(ner, inp, out)

    with gr.Tab("Question answering"):
        question = gr.Textbox(label="Question", value="Where do I work?")
        context = gr.Textbox(label="Context", value=NATE, lines=3)
        out = gr.Textbox(label="Answer")
        gr.Button("Answer").click(answer, [question, context], out)

    with gr.Tab("Summarization"):
        inp = gr.Textbox(label="Text", value=CRISIS, lines=8)
        out = gr.Textbox(label="Summary", lines=4)
        gr.Button("Summarize").click(tasks.summarize, inp, out)

    with gr.Tab("Translation (fr → en)"):
        inp = gr.Textbox(label="French", value="Ce cours est produit par Hugging Face.")
        out = gr.Textbox(label="English")
        gr.Button("Translate").click(tasks.translate, inp, out)

    with gr.Tab("Image classification"):
        inp = gr.Image(label="Image", type="pil")
        out = gr.Label(label="Top classes")
        gr.Button("Classify").click(classify_image, inp, out)

    with gr.Tab("Speech recognition"):
        inp = gr.Audio(label="Record or upload", sources=["microphone", "upload"], type="filepath")
        out = gr.Textbox(label="Transcript", lines=3)
        gr.Button("Transcribe").click(tasks.transcribe, inp, out)

if __name__ == "__main__":
    demo.launch()
