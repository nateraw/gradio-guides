import gradio as gr


def main(text):
    # A gradio function that lowercases text and returns it
    return text.lower()


# A simple gradio app for text
app = gr.Interface(
    main,
    inputs="text",
    outputs="text",
    title="gradio-guides",
    description="A collection of gradio apps from the gradio guides",
)
app.launch(debug=True)

