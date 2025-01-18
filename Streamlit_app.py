import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Title and Description
st.title("Stable Diffusion Image Generator")
st.markdown("Generate images from text prompts using Stable Diffusion.")

# Sidebar Configuration
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select a model:", ["dreamlike-art/dreamlike-diffusion-1.0"])
image_height = st.sidebar.slider("Image Height (px)", 256, 1024, 512, step=64)
image_width = st.sidebar.slider("Image Width (px)", 256, 1024, 512, step=64)

# User Prompt Input
prompt = st.text_input("Enter your text prompt:", "A beautiful sunset over the mountains")

# Generate Images Button
if st.button("Generate Image"):
    # Load the Model
    with st.spinner("Loading the model..."):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_choice, torch_dtype=torch.float32)
            pipe = pipe.to("cpu")  # CPU-only configuration
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()

    # Generate Images
    with st.spinner("Generating images..."):
        try:
            results = pipe(prompt, height=image_height, width=image_width, num_images_per_prompt=1)
            images = results.images
        except Exception as e:
            st.error(f"Error during image generation: {e}")
            st.stop()

    # Display Images
    st.success("Image generation complete!")
    for img in images:
        st.image(img, caption="Generated Image", use_column_width=True)
