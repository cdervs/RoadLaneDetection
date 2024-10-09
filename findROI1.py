import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

#The following script processes all .jpg images in a folder
#, then detects the two prevalent lines in each image
#using the Hough Line Transform.
#It calculates the average coordinates of the intersection points
#and creates and displays a region of interest, while masking the rest of the image.

# Directory containing images

def main():

    path = input("Enter the path of the frames: ")
    # Get a list of all .jpg files in the specified folder
    if not os.path.isfile(path):
        print("The folder does not exist.")
        return False
    jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    if not jpg_files:
        print("No .jpg files found in the specified folder.")
        return False
    
    #path = 'C:\\Users\\chris\\Desktop\\highway\\data\\'

    x_ = []
    y_ = []
    counter = 0

    for i in os.listdir(path):
        if i.endswith('.jpg'):
            fullpath = os.path.join(path, i)
            img = mpimg.imread(fullpath)
        
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
            # Detect lines using Hough Line Transformation
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
            # If no lines were found, skip this image
            if lines is None:
                continue
        
            limit = 0
            intersection_points = []  # Renamed from 'edges' to 'intersection_points' for clarity
        
            for line in lines:
                if limit >= 2:
                    break
                for r, theta in line:
                    # Calculate the x and y coordinates for drawing the lines
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * r
                    y0 = b * r
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                
                    # Draw the detected line on the image
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                    # Store the intersection points
                    limit += 1
                    A = np.array([[x1, 1], [x2, 1]])
                    B = np.array([y1, y2])
                    try:
                        X = np.linalg.inv(A).dot(B)
                        intersection_points.append(X)
                    except np.linalg.LinAlgError:
                        continue
        
            if len(intersection_points) < 2:
                continue  # If fewer than 2 intersection points, skip this image
        
            if abs(intersection_points[0][0] - intersection_points[1][0]) <= 0.001:
                continue
        
            # Calculate intersection x and y using two intersection points
            x = (intersection_points[1][1] - intersection_points[0][1]) / (intersection_points[0][0] - intersection_points[1][0])
            y = (intersection_points[1][0] * intersection_points[0][1] - intersection_points[0][0] * intersection_points[1][1]) / (intersection_points[1][0] - intersection_points[0][0])
        
            # Append x and y values to their respective lists
            x_.append(x)
            y_.append(y)
            counter += 1

    # Calculate average x and y coordinates if any lines were detected
    if counter > 0:
        avgx, avgy = abs(sum(x_)) / counter, abs(sum(y_)) / counter
        print("Average x:", avgx, "Average y:", avgy)
    else:
        print("No lines detected.")
        avgx, avgy = 0, 0

    # Region of Interest (ROI) based on the average y coordinate
    #at the end, it will be applied on a random frame from the folder
    test_image_path = random.choice(jpg_files)
    testim = mpimg.imread(test_image_path)
    testgray = cv2.cvtColor(testim, cv2.COLOR_BGR2GRAY)

    height, width = testgray.shape
    ROI = np.array([[(0, height), (0, avgy), (width, avgy), (width, height)]], dtype=np.int32)

    # Create a blank mask and apply the ROI mask
    blank = np.zeros_like(testgray)
    region_of_interest = cv2.fillPoly(blank, ROI, 255)
    region_of_interest_image = cv2.bitwise_and(testgray, region_of_interest)

    # Display the result
    cv2.imshow('ROI', region_of_interest_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
