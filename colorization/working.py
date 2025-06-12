'''import argparse
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import numpy as np

from colorizers import *

def main():
	st.title("Image Colorization App")

# Upload input image
	uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	if uploaded_file is not None:
        # Read uploaded image
		#image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
		#try:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            #parser = argparse.ArgumentParser()
            #parser.add_argument('-i','--img_path', type=str, default=image)
            #parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
            #parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
            #opt = parser.parse_args()
                # load colorizers
            colorizer_eccv16 = eccv16(pretrained=True).eval()
            colorizer_siggraph17 = siggraph17(pretrained=True).eval()
            if(1):
                    colorizer_eccv16.cuda()
                    colorizer_siggraph17.cuda()

                # default size to process images is 256x256
                # grab L channel in both original ("orig") and resized ("rs") resolutions
                    img = load_img()
                    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
                    if(opt.use_gpu):
                        tens_l_rs = tens_l_rs.cuda()

                # colorizer outputs 256x256 ab map
                # resize and concatenate to original L channel
                    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
                    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
                    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

                    plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
                    plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

                    plt.figure(figsize=(12,8))
                    plt.subplot(2,2,1)
                    plt.imshow(img)
                    plt.title('Original')
                    plt.axis('off')

                    plt.subplot(2,2,2)
                    plt.imshow(img_bw)
                    plt.title('Input')
                    plt.axis('off')

                    plt.subplot(2,2,3)
                    plt.imshow(out_img_eccv16)
                    plt.title('Output (ECCV 16)')
                    plt.axis('off')

                    plt.subplot(2,2,4)
                    plt.imshow(out_img_siggraph17)
                    plt.title('Output (SIGGRAPH 17)')
                    plt.axis('off')
                    plt.show()
        #except Exception as e:
        #            st.error(f"Error processing the file: {e}")
        #            return
if __name__ == "__main__":
    main()'''

import streamlit as st
import torch
import cv2
import numpy as np
from colorizers import *  # Assuming colorizers is correctly imported

def main():
    st.title("Image Colorization App")

    # Upload input image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Read uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Original Image", use_container_width=True)
            # st.write(f"Image shape: {image_rgb.shape}")

            # Load colorizers
            colorizer_eccv16 = eccv16(pretrained=True).eval()
            colorizer_siggraph17 = siggraph17(pretrained=True).eval()

            # Check GPU availability
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                # st.write("Using GPU for processing")
                colorizer_eccv16.cuda()
                colorizer_siggraph17.cuda()
            else:
                st.write("Using CPU for processing")

            # Process the image
            img = image_rgb
            tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
            # st.write(f"Original L Tensor Shape: {tens_l_orig.shape}")
            # st.write(f"Resized L Tensor Shape: {tens_l_rs.shape}")

            if use_gpu:
                tens_l_rs = tens_l_rs.cuda()

            # Colorize the image
            out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
            out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

            st.write("ECCV16 Output Shape:", out_img_eccv16.shape)
            st.write("SIGGRAPH17 Output Shape:", out_img_siggraph17.shape)

            # Display the colorized images
            st.subheader("Colorized Images")
            st.image(out_img_eccv16, caption="ECCV 16 Output", use_container_width=True)
            st.image(out_img_siggraph17, caption="SIGGRAPH 17 Output", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing the file: {e}")

if __name__ == "__main__":
    main()

