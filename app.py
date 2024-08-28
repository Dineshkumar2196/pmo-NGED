import fitz  # PyMuPDF
import re
import cv2
import numpy as np
import streamlit as st
from io import BytesIO

# Function to find specific details using regular expressions
def find_detail(text, pattern, group=1):
    match = re.search(pattern, text)
    return match.group(group) if match else "Not found"

# Function to find white text and its bounding box
def find_white_text(page):
    text_instances = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    # Checking if the text color is white (1.0 for R, G, B in PDF context)
                    if span["color"] == 1.0:
                        text_instances.append({
                            "text": span["text"],
                            "bbox": span["bbox"]
                        })
    return text_instances

st.title("Networking Processing App")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the PDF file
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    filename = uploaded_file.name

    # Combined mask for all pages
    combined_mask_all_pages = None
    img_with_bbox_all_pages = None

    # Patterns to search for specific details
    patterns = {
        "Date Requested": r'Date Requested:\s*(\S+)',
        "Job Reference": r'Job Reference:\s*(\S+)',
        "Site Location": r'Site Location:\s*([\d\s]+)',
        "Your Scheme/Reference": r'Your Scheme/Reference:\s*(\S+)',
        "Gas Warning": r'WARNING! This area contains (.*)'
    }

    # Loop through all pages in the PDF
    for page_number in range(len(pdf_document)):
        st.write(f"\nProcessing Page {page_number + 1}")

        # Load the page
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()

        # Handle different numbers of color channels
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Convert grayscale or RGBA to RGB
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Extract text from the page
        text = page.get_text()

        # Extract and display the details
        details = {key: find_detail(text, pattern) for key, pattern in patterns.items()}
        for key, value in details.items():
            st.write(f"{key}: {value}")

        # Find white text
        white_texts = find_white_text(page)

        # Draw bounding boxes around white text
        for white_text in white_texts:
            x_min, y_min, x_max, y_max = map(int, white_text["bbox"])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the range of the purple color in HSV
        lower_purple = np.array([140, 50, 50])
        upper_purple = np.array([160, 255, 255])

        # Create a mask for the purple color
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

        # Define the range of green color in HSV
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([80, 255, 255])

        # Create a mask for the green color
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Define the range for dark yellow-orange color in HSV
        lower_yellow_orange = np.array([20, 180, 180])
        upper_yellow_orange = np.array([30, 255, 255])

        # Create a mask for the dark yellow-orange color
        mask_orange = cv2.inRange(hsv, lower_yellow_orange, upper_yellow_orange)

        # Combine purple with each specified color
        combined_masks = {
            "Hand symbol": cv2.bitwise_or(mask_green, mask_purple),
            "Target": cv2.bitwise_or(mask_orange, mask_purple),
        }

        # Manually define the bounding box coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = 8, 10, 585, 580

        # Apply bounding box mask to each combined mask
        combined_mask_page = np.zeros_like(mask_purple)
        for combined_mask in combined_masks.values():
            bbox_mask = np.zeros_like(combined_mask)
            bbox_mask[y_min:y_max, x_min:x_max] = combined_mask[y_min:y_max, x_min:x_max]
            combined_mask_page = cv2.bitwise_or(combined_mask_page, bbox_mask)

        # Accumulate the masks across all pages
        if combined_mask_all_pages is None:
            combined_mask_all_pages = combined_mask_page
        else:
            combined_mask_all_pages = cv2.bitwise_or(combined_mask_all_pages, combined_mask_page)

        # Apply the bounding box mask to the original image for this page
        result_page = cv2.bitwise_and(img, img, mask=combined_mask_page)

        # Draw the bounding box on the result image for this page
        result_with_bbox_page = result_page.copy()
        cv2.rectangle(result_with_bbox_page, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Combine this page's result with the overall result image
        if img_with_bbox_all_pages is None:
            img_with_bbox_all_pages = result_with_bbox_page
        else:
            img_with_bbox_all_pages = cv2.addWeighted(img_with_bbox_all_pages, 1, result_with_bbox_page, 1, 0)

    # Display the overall combined result with bounding boxes
    if img_with_bbox_all_pages is not None:
        st.image(cv2.cvtColor(img_with_bbox_all_pages, cv2.COLOR_BGR2RGB), caption='EMBEDDED NETWORK', use_column_width=True)
