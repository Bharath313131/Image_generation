# pip install diffusers transformers accelerate
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Title and Introduction
st.title("Stable Diffusion Image Generator")
st.markdown("""
Generate stunning images from text prompts using Stable Diffusion models.
""")

# Model Selection
model_option = st.selectbox(
    "Choose a model:",
    ["dreamlike-art/dreamlike-diffusion-1.0", "stabilityai/stable-diffusion-xl-base-1.0"]
)

# Prompt Input
prompt = st.text_input("Enter your text prompt:", "A serene landscape with mountains and a lake")

# Image Configuration
st.sidebar.header("Image Configuration")
height = st.sidebar.number_input("Image Height", min_value=256, max_value=1024, value=512, step=64)
width = st.sidebar.number_input("Image Width", min_value=256, max_value=1024, value=512, step=64)
num_images = st.sidebar.slider("Number of Images", min_value=1, max_value=4, value=1)

# Generate Button
if st.button("Generate Images"):
    # Load Model
    with st.spinner("Loading the model... This may take a few seconds."):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_option,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe = pipe.to("cpu")

    # Generate Images
    with st.spinner("Generating images..."):
        result_images = pipe(prompt, height=height, width=width, num_images_per_prompt=num_images).images

    # Display Images
    st.header("Generated Images")
    for img in result_images:
        st.image(img, use_column_width=True)
