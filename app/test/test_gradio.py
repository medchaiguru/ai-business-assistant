import gradio as gr

def greet(name):
    return f"Hello, {name}!"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            name_input = gr.Textbox(
                label="Your name",
                placeholder="Type your name and press Enter..."
            )
        with gr.Column(scale=1):
            btn = gr.Button("Submit")

    output = gr.Textbox(label="Greeting")

    # Works even though it's inside Row/Column
    name_input.submit(fn=greet, inputs=name_input, outputs=output)
    btn.click(fn=greet, inputs=name_input, outputs=output)

demo.launch()
