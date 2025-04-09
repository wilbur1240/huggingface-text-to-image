import gradio as gr

def infer(prompt):
    return f"Hello from: {prompt}"

with gr.Blocks() as demo:
    txt = gr.Textbox()
    out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(infer, txt, out)

if __name__ == "__main__":
    demo.launch(show_api=False, prevent_thread_lock=True)
