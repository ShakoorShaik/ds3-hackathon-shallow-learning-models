
import bm3d
import pandas as pd
import numpy as np
import cv2  
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def highmask(image, weight):
    # Calculate threshold
    flattened_image = image.flatten()
    flattened_image.sort()

    threshold_value = max(flattened_image) - (max(flattened_image) - min(flattened_image)) * weight
    
    # Apply threshold
    ret, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply mask to original image
    masked_image = cv2.bitwise_and(image, image, mask=thresholded)
    return masked_image

def contrastadd(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def canny_edge(image, lower, upper):
    tight = cv2.Canny(image, lower, upper)
    # show the output Canny edge maps
    return tight

def gaussian_blur(image, weight):
    return cv2.GaussianBlur(image, (weight, weight), 0)

def preprocess_image(image_path, imgclass):
    img = cv2.imread(image_path)  # Load image in rgb
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image into grayscale
    # invert image
    #denoised_image = bm3d.bm3d(gaussian_img, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
#    return canny_edge(contrastadd(gaussian_img))
    if imgclass == 0:
        img = contrastadd(img)
        img = highmask(img, 0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.5)
        return img
    elif imgclass == 1:
        # edge detection
        img = contrastadd(img)
        img = highmask(img, 0.8)
        # remove the artifacts
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.8)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.7)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.4)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.4)

        return img
    elif imgclass == 2:
        img = contrastadd(img)
        img = highmask(img,0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img,0.5)
        return img
    elif imgclass == 3:
        # edge detection
        img = contrastadd(img)
        img = highmask(img, 0.8)
        # remove the artifacts
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.8)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.7)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.4)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.4)
        return img
    elif imgclass == 4:
        img = contrastadd(img)
        img = highmask(img, 0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.5)
        return img
    else:
        img = contrastadd(img)
        img = highmask(img, 0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.5)
        return img


# Load training data
df_train = pd.read_csv("fungi_train.csv")
df_test = pd.read_csv("fungi_test.csv")

for i, row in df_train.iterrows():
    print(row["Path"])
    img = preprocess_image(row["Path"], row["ClassId"])
    cv2.imwrite(f"grayscale/{row.ClassId}/{i}.png", img)

for i, row in df_test.iterrows():
    print(row["Path"])
    img = preprocess_image(row["Path"], "test")
    cv2.imwrite(f"grayscale/test/{i}.png", img)

