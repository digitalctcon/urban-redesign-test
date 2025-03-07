import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline
from streamlit_drawable_canvas import st_canvas
import os
from datetime import datetime
import csv

# Set Streamlit layout
st.set_page_config(layout="wide")

# ---- Load Stable Diffusion Model ----
@st.cache_resource
def load_model():
    """Loads the Stable Diffusion Inpainting model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=dtype
    ).to(device)
    return pipe

pipe = load_model()

# ---- Session State Handling ----
def on_upload():
    """Handles image upload and ensures session state is updated."""
    if "input_image" in st.session_state and st.session_state["input_image"] is not None:
        image = Image.open(st.session_state["input_image"]).convert("RGB").resize((512, 512))
        st.session_state["initial_image"] = image
        # Reset mask and output when new image is uploaded
        for key in ["mask", "output_image"]:
            st.session_state.pop(key, None)

def on_select_image():
    """Handles image selection from predefined images."""
    if "selected_image" in st.session_state and st.session_state["selected_image"] is not None:
        image_path = os.path.join(st.session_state["image_folder"], st.session_state["selected_image"])
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        st.session_state["initial_image"] = image
        # Reset mask and output when new image is selected
        for key in ["mask", "output_image"]:
            st.session_state.pop(key, None)

def make_canvas_dict(brush, paint_mode, _reset_state):
    """Creates a dictionary for the drawing canvas."""
    return dict(
        fill_color="white",
        stroke_color="white",
        background_color="#FFFFFF",
        background_image=st.session_state.get("initial_image", None),
        stroke_width=brush,
        initial_drawing={"version": "4.4.0", "objects": []} if _reset_state else None,
        update_streamlit=True,
        height=512,
        width=512,
        drawing_mode=paint_mode,
        key="canvas",
    )

# ---- Sidebar UI ----
def make_sidebar():
    with st.sidebar:
        st.write("## 📂 Upload or Select an Image")
        uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="input_image", on_change=on_upload)

        st.write("### Or select from existing images")
        image_category = st.selectbox("Select Category", ["park", "road", "street"])
        image_folder = os.path.join("data", "images", image_category)
        st.session_state["image_folder"] = image_folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg')) and "redesign" not in f]

        # Display image previews
        selected_image = None
        cols = st.columns(3)
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            display_name = os.path.splitext(image_file)[0].replace("_", " ").title()
            with cols[i % 3]:
                if st.button(display_name, key=image_file):
                    selected_image = image_file
                    st.session_state["selected_image"] = selected_image
                    on_select_image()

        st.write("## ✏️ Drawing Settings")
        paint_mode = "freedraw"  # Fixed mode to freedraw
        brush = st.slider("✏️ Brush Size", 5, 50, 20)

        st.write("## 🎨 Inpainting Settings")
        prompt = st.text_input("Describe the redesign", "A walkway made of photorealistic green grass, well-maintained.")

    return uploaded_image, paint_mode, brush, prompt

# ---- Main App UI ----
def get_next_generation_id(log_file):
    """Gets the next generation ID based on the existing log file."""
    if not os.path.isfile(log_file):
        return 1
    with open(log_file, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=';')
        rows = list(reader)
        if len(rows) < 2:
            return 1
        last_row = rows[-1]
        return int(last_row[0]) + 1

def log_generation(input_path, output_path, prompt, image_type, category):
    """Logs the generation details to a CSV file."""
    log_file = os.path.join("data", "generations", "generation_log.csv")
    generation_id = get_next_generation_id(log_file)
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["Generation ID", "Category", "Image Type", "Prompt", "Input File", "Output File", "Datetime"])
        writer.writerow([generation_id, category, image_type, prompt, input_path, output_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def main():
    st.title("Urban Redesigner")

    # Get sidebar inputs
    uploaded_image, paint_mode, brush, prompt = make_sidebar()

    # Ensure an image has been uploaded
    if "initial_image" not in st.session_state:
        st.warning("📂 Please upload an image to continue.")
        return  # Stop execution if no image is uploaded

    # ---- Drawing Canvas ----
    canvas_dict = make_canvas_dict(brush, paint_mode, False)
    canvas_result = st_canvas(**canvas_dict)

    # Convert mask to usable format
    if canvas_result.image_data is not None:
        mask = Image.fromarray((canvas_result.image_data[:, :, 0] > 200).astype(np.uint8) * 255)
        st.session_state["mask"] = mask

    # ---- Generate New Design ----
    if st.button("✨ Generate New Design") and "mask" in st.session_state:
        st.write("🚀 Generating new urban design...")

        output_image = pipe(
            prompt=prompt,
            image=st.session_state["initial_image"],
            mask_image=st.session_state["mask"]
        ).images[0]

        st.session_state["output_image"] = output_image

        # Save the new image in the same folder as the original image
        if "selected_image" in st.session_state:
            original_image_path = os.path.join(st.session_state["image_folder"], st.session_state["selected_image"])
            base_name, ext = os.path.splitext(st.session_state["selected_image"])
            timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
            new_image_name = f"{base_name}_redesign_{timestamp}{ext}"
            new_image_path = os.path.join(st.session_state["image_folder"], new_image_name)
            st.session_state["output_image"].save(new_image_path)
            log_generation(original_image_path, new_image_path, prompt, "from_templates", os.path.basename(st.session_state["image_folder"]))  # Log the generation
        else:
            new_image_path = "urban_redesign.png"
            st.session_state["output_image"].save(new_image_path)
            log_generation("uploaded_image", new_image_path, prompt, "uploaded_by_user", "unknown")  # Log the generation

        st.download_button(
            "💾 Download New Image",
            data=open(new_image_path, "rb"),
            file_name=os.path.basename(new_image_path),
            mime="image/png"
        )

    # ---- Show Results ----
    if "output_image" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state["initial_image"], caption="🏙️ Original Image", use_container_width=True)
        with col2:
            st.image(st.session_state["output_image"], caption="🏗️ Redesigned Urban Space", use_container_width=True)

if __name__ == "__main__":
    main()