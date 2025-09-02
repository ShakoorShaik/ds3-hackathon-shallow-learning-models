import pandas as pd
import numpy as np
import cv2  
from PIL import Image, ImageTk  # Import PIL for PNG support
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tkinter as tk

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


currentImgPath = ""

def tossfunc(window):
    kill_list.append(currentImgPath)
    window.destroy()
def keepfunc(window):
    window.destroy()

def showandask():
    window = tk.Tk()
    label = tk.Label(window, text=f"current path: {currentImgPath}")
    label.grid(column=0, row=1)
    img1 = ImageTk.PhotoImage(Image.open("original.jpg"))
    btna = tk.Button(window, image=img1, command=None)
    btna.grid(column=0,row=0)
    img2 = ImageTk.PhotoImage(Image.open("gaussian_img.jpg"))
    btnb = tk.Button(window, image=img2, command=None)
    btnb.grid(column=1,row=0)
    img3 = ImageTk.PhotoImage(Image.open("bnw_img.jpg"))
    btnb = tk.Button(window, image=img3, command=None)
    btnb.grid(column=2,row=0)
    img4 = ImageTk.PhotoImage(Image.open("high_img.jpg"))
    btnb = tk.Button(window, image=img4, command=None)
    btnb.grid(column=3,row=0)
    img5 = ImageTk.PhotoImage(Image.open("preprocessed_img.jpg"))
    btnc = tk.Button(window, image=img5, command=None)
    btnc.grid(column=4,row=0)
    keep = tk.Button(window, text="keep", command=lambda: keepfunc(window))
    keep.grid(column=0, row=2)
    toss = tk.Button(window, text="toss", command=lambda : tossfunc(window)) 
    toss.grid(column=1, row=2)
    window.mainloop()

highesti = 1014
kill_list = ['Train/0/0_0004.jpg', 'Train/0/0_0007.jpg', 'Train/0/0_0018.jpg', 'Train/0/0_0021.jpg', 'Train/0/0_0022.jpg', 'Train/0/0_0023.jpg', 'Train/0/0_0024.jpg', 'Train/0/0_0026.jpg', 'Train/0/0_0027.jpg', 'Train/0/0_0032.jpg', 'Train/0/0_0034.jpg', 'Train/0/0_0035.jpg', 'Train/0/0_0037.jpg', 'Train/0/0_0061.jpg', 'Train/0/0_0063.jpg', 'Train/0/0_0066.jpg', 'Train/0/0_0067.jpg', 'Train/0/0_0068.jpg', 'Train/0/0_0069.jpg', 'Train/0/0_0070.jpg', 'Train/0/0_0083.jpg', 'Train/0/0_0084.jpg', 'Train/0/0_0089.jpg', 'Train/0/0_0095.jpg', 'Train/0/0_0144.jpg', 'Train/0/0_0145.jpg', 'Train/0/0_0146.jpg', 'Train/0/0_0179.jpg', 'Train/0/0_0181.jpg', 'Train/0/0_0202.jpg', 'Train/0/0_0205.jpg', 'Train/0/0_0209.jpg', 'Train/0/0_0210.jpg', 'Train/0/0_0215.jpg', 'Train/0/0_0218.jpg', 'Train/0/0_0220.jpg', 'Train/0/0_0221.jpg', 'Train/0/0_0222.jpg', 'Train/0/0_0223.jpg', 'Train/0/0_0227.jpg', 'Train/0/0_0229.jpg', 'Train/0/0_0230.jpg', 'Train/0/0_0231.jpg', 'Train/0/0_0234.jpg', 'Train/0/0_0235.jpg', 'Train/0/0_0239.jpg', 'Train/0/0_0241.jpg', 'Train/0/0_0245.jpg', 'Train/0/0_0248.jpg', 'Train/0/0_0266.jpg', 'Train/0/0_0267.jpg', 'Train/0/0_0284.jpg', 'Train/0/0_0286.jpg', 'Train/0/0_0291.jpg', 'Train/0/0_0293.jpg', 'Train/0/0_0294.jpg', 'Train/0/0_0297.jpg', 'Train/0/0_0298.jpg', 'Train/0/0_0299.jpg', 'Train/0/0_0301.jpg', 'Train/0/0_0302.jpg', 'Train/0/0_0303.jpg', 'Train/0/0_0304.jpg', 'Train/0/0_0305.jpg', 'Train/0/0_0311.jpg', 'Train/0/0_0312.jpg', 'Train/0/0_0320.jpg', 'Train/0/0_0328.jpg', 'Train/0/0_0329.jpg', 'Train/0/0_0330.jpg', 'Train/0/0_0331.jpg', 'Train/0/0_0333.jpg', 'Train/0/0_0334.jpg', 'Train/0/0_0337.jpg', 'Train/0/0_0341.jpg', 'Train/0/0_0342.jpg', 'Train/0/0_0347.jpg', 'Train/0/0_0352.jpg', 'Train/0/0_0354.jpg', 'Train/0/0_0358.jpg', 'Train/0/0_0363.jpg', 'Train/0/0_0370.jpg', 'Train/0/0_0371.jpg', 'Train/0/0_0376.jpg', 'Train/0/0_0378.jpg', 'Train/0/0_0381.jpg', 'Train/0/0_0382.jpg', 'Train/0/0_0383.jpg', 'Train/0/0_0384.jpg', 'Train/0/0_0385.jpg', 'Train/0/0_0388.jpg', 'Train/0/0_0389.jpg', 'Train/0/0_0399.jpg', 'Train/0/0_0405.jpg', 'Train/0/0_0407.jpg', 'Train/0/0_0408.jpg', 'Train/0/0_0412.jpg', 'Train/0/0_0413.jpg', 'Train/0/0_0415.jpg', 'Train/0/0_0416.jpg', 'Train/0/0_0417.jpg', 'Train/0/0_0426.jpg', 'Train/0/0_0429.jpg', 'Train/0/0_0430.jpg', 'Train/0/0_0435.jpg', 'Train/0/0_0439.jpg', 'Train/0/0_0442.jpg', 'Train/0/0_0443.jpg', 'Train/0/0_0444.jpg', 'Train/0/0_0446.jpg', 'Train/0/0_0447.jpg', 'Train/0/0_0448.jpg', 'Train/0/0_0449.jpg', 'Train/0/0_0450.jpg', 'Train/0/0_0451.jpg', 'Train/0/0_0452.jpg', 'Train/0/0_0453.jpg', 'Train/0/0_0456.jpg', 'Train/0/0_0457.jpg', 'Train/0/0_0458.jpg', 'Train/0/0_0459.jpg', 'Train/0/0_0462.jpg', 'Train/0/0_0467.jpg', 'Train/0/0_0473.jpg', 'Train/0/0_0485.jpg', 'Train/0/0_0486.jpg', 'Train/0/0_0487.jpg', 'Train/0/0_0494.jpg', 'Train/0/0_0504.jpg', 'Train/0/0_0508.jpg', 'Train/0/0_0512.jpg', 'Train/0/0_0516.jpg', 'Train/0/0_0518.jpg', 'Train/0/0_0519.jpg', 'Train/0/0_0523.jpg', 'Train/0/0_0524.jpg', 'Train/0/0_0529.jpg', 'Train/0/0_0534.jpg', 'Train/0/0_0535.jpg', 'Train/0/0_0537.jpg', 'Train/0/0_0558.jpg', 'Train/0/0_0561.jpg', 'Train/0/0_0566.jpg', 'Train/0/0_0567.jpg', 'Train/0/0_0576.jpg', 'Train/0/0_0579.jpg', 'Train/0/0_0580.jpg', 'Train/0/0_0587.jpg', 'Train/0/0_0590.jpg', 'Train/0/0_0593.jpg', 'Train/0/0_0602.jpg', 'Train/0/0_0603.jpg', 'Train/0/0_0608.jpg', 'Train/0/0_0609.jpg', 'Train/0/0_0611.jpg', 'Train/0/0_0612.jpg', 'Train/0/0_0614.jpg', 'Train/0/0_0623.jpg', 'Train/0/0_0625.jpg', 'Train/0/0_0626.jpg', 'Train/0/0_0631.jpg', 'Train/0/0_0634.jpg', 'Train/0/0_0645.jpg', 'Train/0/0_0650.jpg', 'Train/0/0_0651.jpg', 'Train/0/0_0686.jpg', 'Train/0/0_0687.jpg', 'Train/0/0_0688.jpg', 'Train/0/0_0689.jpg', 'Train/0/0_0691.jpg', 'Train/0/0_0698.jpg', 'Train/0/0_0707.jpg', 'Train/0/0_0710.jpg', 'Train/0/0_0738.jpg', 'Train/0/0_0775.jpg', 'Train/0/0_0776.jpg', 'Train/0/0_0788.jpg', 'Train/0/0_0790.jpg', 'Train/0/0_0791.jpg', 'Train/0/0_0794.jpg', 'Train/0/0_0795.jpg', 'Train/0/0_0796.jpg', 'Train/0/0_0797.jpg', 'Train/0/0_0800.jpg', 'Train/0/0_0801.jpg', 'Train/0/0_0802.jpg', 'Train/0/0_0804.jpg', 'Train/0/0_0810.jpg', 'Train/0/0_0835.jpg', 'Train/0/0_0838.jpg', 'Train/0/0_0846.jpg', 'Train/0/0_0847.jpg', 'Train/0/0_0852.jpg', 'Train/0/0_0857.jpg', 'Train/0/0_0861.jpg', 'Train/0/0_0862.jpg', 'Train/0/0_0863.jpg', 'Train/0/0_0864.jpg', 'Train/0/0_0865.jpg', 'Train/0/0_0883.jpg', 'Train/0/0_0884.jpg', 'Train/0/0_0885.jpg', 'Train/0/0_0886.jpg', 'Train/0/0_0887.jpg', 'Train/0/0_0888.jpg', 'Train/0/0_0889.jpg', 'Train/0/0_0900.jpg', 'Train/0/0_0907.jpg', 'Train/0/0_0912.jpg', 'Train/0/0_0913.jpg', 'Train/0/0_0929.jpg', 'Train/0/0_0930.jpg', 'Train/0/0_0937.jpg', 'Train/0/0_0940.jpg', 'Train/0/0_0945.jpg', 'Train/0/0_0950.jpg', 'Train/0/0_0965.jpg', 'Train/0/0_0966.jpg', 'Train/0/0_0967.jpg', 'Train/0/0_0975.jpg', 'Train/1/1_0001.jpg', 'Train/1/1_0003.jpg', 'Train/1/1_0004.jpg', 'Train/1/1_0005.jpg', 'Train/1/1_0006.jpg', 'Train/1/1_0007.jpg', 'Train/1/1_0009.jpg', 'Train/1/1_0010.jpg', 'Train/1/1_0012.jpg']


for i, row in df_train.iterrows():
    if i < highesti:
        continue
    print(row["Path"])
    currentImgPath = row["Path"]
    original_img = cv2.imread(row["Path"])
    gaussian_img = gaussian_blur(original_img, 3)
    bnw_img = contrastadd(cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2GRAY))
    high_img = highmask(bnw_img, 0.8)
    preprocessed_img = preprocess_image(row["Path"], row["ClassId"])
    cv2.imwrite("original.jpg", original_img)
    cv2.imwrite("gaussian_img.jpg", gaussian_img)
    cv2.imwrite("bnw_img.jpg", bnw_img)
    cv2.imwrite("high_img.jpg", bnw_img)
    cv2.imwrite("preprocessed_img.jpg", preprocessed_img)
    showandask()    
    print(f"kill_list: {kill_list}")
    
    # tkinter prompt 
