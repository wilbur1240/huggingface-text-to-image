import gradio as gr
import numpy as np
import random
from huggingface_hub import InferenceClient
import os

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_API_TOKEN"]
)

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    image = client.text_to_image(
        prompt=prompt,
        model="stabilityai/sdxl-turbo",
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        seed=seed,
    )
    return image, seed

# Step 1: Build the interface (as before)
interface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt", value=""),
        gr.Slider(0, MAX_SEED, label="Seed", value=0),
        gr.Checkbox(label="Randomize Seed", value=True),
        gr.Slider(256, MAX_IMAGE_SIZE, label="Width", step=32, value=768),
        gr.Slider(256, MAX_IMAGE_SIZE, label="Height", step=32, value=768),
        gr.Slider(0, 10, label="Guidance Scale", step=0.1, value=0.0),
        gr.Slider(1, 50, label="Steps", step=1, value=2)
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Number(label="Seed Used")
    ],
    examples=[
        ["A futuristic cityscape at sunset", "", 0, True, 768, 768, 0.0, 2],
        ["A cat in a space suit", "", 0, True, 768, 768, 0.0, 2],
    ],
    allow_flagging="never"
)

# Step 2: Render the interface inside a Blocks container
with gr.Blocks() as demo:
    interface.render()

# Step 3: Launch cleanly
if __name__ == "__main__":
    demo.launch(show_api=False)
