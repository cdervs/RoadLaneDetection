import os
import cv2
import numpy as np

#The following script detects the largest contour in multiple images from a specified folder
#computes the ROI based on the height of the largest contour in each image
#and then calculates the average height of those ROIs across all images.

def process_image_for_largest_contour(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Image {image_path} not found or unable to load.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to create a binary image
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Dilate the image to fill gaps in contours
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Return None if no contours are found
    if len(contours) == 0:
        return None
    
    # Find the largest contour by area
    maxim = 0
    main_h = 0
    
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if cv2.contourArea(cntr) > maxim:
            maxim = cv2.contourArea(cntr)
            main_h = h  # Height of the bounding box of the largest contour
    
    return main_h

def process_images_in_folder(folder_path):
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    
    if len(jpg_files) == 0:
        print("No .jpg files found in the folder.")
        return
    
    total_height = 0
    count = 0
    
    for img_file in jpg_files:
        fullpath = os.path.join(folder_path, img_file)
        main_h = process_image_for_largest_contour(fullpath)
        
        if main_h is not None:
            total_height += main_h
            count += 1
            print(f"Processed {img_file}: Largest contour height = {main_h}")
    
    if count > 0:
        avg_height = total_height / count
        print(f"Average height of largest contours across {count} images: {avg_height}")
    else:
        print("No valid contours found in any image.")


def main():
    # Set the folder path containing images
    folder_path = input("Enter the images' folder path: ")

    # Process all images in the folder and calculate average contour height
    process_images_in_folder(folder_path)
