import gradio as gr

def infer(prompt):
    return f"Hello from: {prompt}"

demo = gr.Interface(
    fn=infer,
    inputs=gr.Textbox(label="Enter something"),
    outputs=gr.Textbox(label="Output")
)

demo.launch(show_api=False)