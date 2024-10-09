import os
import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.image as mpimg
import random
from skimage.color import rgb2gray

# Define a list of common video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return out

def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

def add_sp(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # Getting the dimensions of the image
    row , col = img.shape
          
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(3000, 50000)
    for i in range(number_of_pixels):
            
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
              
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
              
        # Color that pixel to white
        img[y_coord][x_coord] = 255
              
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
            
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
              
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        
        # Color that pixel to black
        img[y_coord][x_coord] = 0
        
    return img

def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    image = mat2gray(image)
    
    mode = mode.lower()
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)
        
    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)        
        out = image  + noise
    if clip:        
        out = np.clip(out, low_clip, 1.0)
        
    return out

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
            
    return data_final

def extract(noise_type = 'no'):
	
    videopath = input("Enter the path of the video you want to extract the noise from: ")
    if not os.path.isfile(videopath):
        print("The file does not exist.")
        return False
    # Check the file extension
    _, ext = os.path.splitext(videopath)
    if ext.lower() not in VIDEO_EXTENSIONS:
        print(f"The file is not a recognized video format (extension: {ext}).")
        return False
    
    cam = cv2.VideoCapture(videopath) 
    #cam = cv2.VideoCapture('C:\\Users\\chris\\Downloads\\part2\\april21.avi')
    #path = 'C:\\Users\\chris\\Desktop\\highway\\data'
    
    # Get the Desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    

    
    extractednoisepath = "extracted_noise_frames"
    currentpath = os.path.join(desktop_path,extractednoisepath)
    
    try :
            
        if not os.path.exists(currentpath):
            os.makedirs(currentpath)
                
    except OSError:
        print ('Error: Creating directory of data')
        
    currentframe = 0
        
    while(True):
            
        ret,frame = cam.read()
             
        if ret:
                 
            #name = 'C:/Users/chris/Desktop/highway/data/frame' + str(currentframe) + '.jpg'
            name = os.path.join(currentpath, 'frame', str(currentframe), '.jpg')
            if noise_type == 'gaussian':
                frame = random_noise(frame,'gaussian', mean=0,var=0.1)
                frame = np.uint8(frame*255)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #name = 'C:/Users/chris/Desktop/highway/gaussdata/gaussian' + str(currentframe) + '.jpg'
                
                kernel = gaussian_kernel(3)
                frame = wiener_filter(frame, kernel, K = 0.5)
                #name = 'C:/Users/chris/Desktop/highway/wienerdata/wiener' + str(currentframe) + '.jpg'
            elif noise_type == 's&p':
                frame = add_sp(frame)
                #name = 'C:/Users/chris/Desktop/highway/s&pdata/s&p' + str(currentframe) + '.jpg'
                
                arr = np.array(frame)
                frame = median_filter(arr, 3)
                #name = 'C:/Users/chris/Desktop/highway/mediandata/median' + str(currentframe) + '.jpg'
                 
            cv2.imwrite(name, frame)
                 
            currentframe += 1
                 
        else:
            break
        
    cam.release()
    cv2.destroyAllWindows()
    


def main():
    
    extract('gaussian')
    
if __name__ == "__main__":
    main()