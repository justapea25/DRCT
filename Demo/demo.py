# Import necessary libraries
import argparse
import cv2
import numpy as np
import os
import streamlit as st
import torch
from drct.archs.DRCT_arch import DRCT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained DRCT model
def load_model(model_path):
    model = DRCT(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
                  squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.0,
                  depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, gc=32,
                  mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model.to(device)
    return model

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)
    return img

# Function to postprocess the output image
def postprocess_image(output):
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # Convert from RGB back to BGR for correct color
    return (output * 255.0).round().astype(np.uint8)

# Super-resolve the image
def super_resolve(model, low_res_image):
    with torch.no_grad():
        img = preprocess_image(low_res_image)
        output = model(img)
        high_res_image = postprocess_image(output)
    return high_res_image

import streamlit as st
import cv2
import numpy as np

# Streamlit UI
def main():
    st.title("Image Super-Resolution using DRCT Model")

    # Upload low-resolution image
    uploaded_file = st.file_uploader("Upload a low-resolution image", type=["jpg", "jpeg", "png"])

    # Check if the image was loaded successfully
    if uploaded_file is not None:
        # Decode the uploaded image
        low_res_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Check if the image was loaded successfully
        if low_res_image is None:
            st.error("Failed to load the image. Please upload a valid image file.")
        else:
            # Convert from BGR to RGB format
            low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)

            # Create two columns
            col1, col2 = st.columns(2)

            # Display the low-resolution image in the first column
            with col1:
                st.image(low_res_image, caption='Low Resolution Image', width=300)

            # Load the model
            model = load_model("./experiments/pretrained_models/DRCT_SRx4_ImageNet-pretrain.pth")

            # Process the image
            if st.button("Super-Resolve"):
                # Super-resolve the image
                high_res_image = super_resolve(model, low_res_image)

                # Display the high-resolution image in the second column
                with col2:
                    st.image(high_res_image, caption='High Resolution Image', width=300)

if __name__ == '__main__':
    main()

