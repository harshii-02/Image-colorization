
import argparse
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
		try:
			image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
		except Exception as e:
			st.error(f"Error processing the file: {e}")
			return
		# load colorizers
		colorizer_eccv16 = eccv16(pretrained=True).eval()
		colorizer_siggraph17 = siggraph17(pretrained=True).eval()
		st.image(image, caption='Uploaded Image', use_column_width=True)

        # Colorize the image using your model (define this function based on your model)
        #colorized_image = colorize_image(image)  # Implement your colorization model here
        #st.image(colorized_image, caption='Colorized Image', use_column_width=True)

		colorizer_eccv16.cuda()
		colorizer_siggraph17.cuda()
		img = uploaded_file
		#parser = argparse.ArgumentParser()
		#parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
		#parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
		#parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
		#opt = parser.parse_args()

		#if(opt.use_gpu):
		#	colorizer_eccv16.cuda()
		#	colorizer_siggraph17.cuda()

		# default size to process images is 256x256
		# grab L channel in both original ("orig") and resized ("rs") resolutions
		#img = load_img(opt.img_path)
		#img = uploaded_file

		#(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
		#if(opt.use_gpu):
		#	tens_l_rs = tens_l_rs.cuda()

		# colorizer outputs 256x256 ab map
		# resize and concatenate to original L channel
		#img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))

if __name__ == "__main__":
    main()