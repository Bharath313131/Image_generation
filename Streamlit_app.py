import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Title and Description
st.title("Stable Diffusion Image Generator (Optimized for CPU)")
st.markdown("""
Generate images from text prompts using Stable Diffusion. This app is optimized for CPU setups and includes options for local model loading.
""")

# Sidebar Configuration
st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Select a Stable Diffusion Model:",
    ["dreamlike-art/dreamlike-diffusion-1.0", "stabilityai/stable-diffusion-xl-base-1.0"],
)

local_load = st.sidebar.checkbox("Use locally downloaded models", value=False)
image_height = st.sidebar.slider("Image Height (px)", 256, 1024, 512, step=64)
image_width = st.sidebar.slider("Image Width (px)", 256, 1024, 512, step=64)
num_images = st.sidebar.slider("Number of Images", 1, 2, 1)  # Reduced for performance on CPU.

# User Prompt Input
prompt = st.text_input("Enter your text prompt:", "A futuristic cityscape at sunset")

# Generate Images Button
if st.button("Generate Image"):
    # Load the Model
    with st.spinner("Loading model... This may take some time."):
        try:
            # Load model locally or from Hugging Face
            if local_load:
                model_path = "./model_directory"
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # Use CPU-compatible precision
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_choice,
                    torch_dtype=torch.float32,  # Use CPU precision
                )
            pipe = pipe.to("cpu")  # Ensure it runs on CPU
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()

    # Generate Images
    with st.spinner("Generating images... Please wait."):
        try:
            result = pipe(prompt, height=image_height, width=image_width, num_images_per_prompt=num_images)
            images = result.images
        except Exception as e:
            st.error(f"Error during image generation: {e}")
            st.stop()

    # Display Images
    st.success("Image generation complete!")
    for i, img in enumerate(images):
        st.image(img, caption=f"Generated Image {i+1}", use_column_width=True)
