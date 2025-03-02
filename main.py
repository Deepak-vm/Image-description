import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def fetch_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    else:
        return None

def describe_image(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_length=150, num_beams=7, repetition_penalty=2.5)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# Streamlit UI
st.title("Image Captioning using BLIP")
st.write("Enter an image URL to generate a caption.")

url = st.text_input("Image URL:")
if st.button("Generate Caption"):
    if url:
        img = fetch_image_from_url(url)
        if img:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Generating caption..."):
                description = describe_image(img)
                st.success("Caption generated successfully!")
                st.write("**Description:**", description)
        else:
            st.error("Failed to fetch the image. Please check the URL.")
    else:
        st.warning("Please enter a valid image URL.")
