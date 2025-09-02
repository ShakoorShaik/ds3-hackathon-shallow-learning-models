import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import cv2  # Added for image preprocessing
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

sift = cv2.SIFT_create()

def feature_matching(img):
    kp = sift.detect(img,None)
    img= cv2.drawKeypoints(img, kp, img)
    return img

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
    img = cv2.resize(img, (224, 224))  # Resize image to 224 x 224
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
        img = gaussian_blur(img, 7)
        img = highmask(img, 0.4)
        img = canny_edge(img,250,350)
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
        img = gaussian_blur(img, 7)
        img = highmask(img, 0.4)
        img = canny_edge(img,250,350)
    elif imgclass == 2:
        img = contrastadd(img)
        img = highmask(img,0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img,0.5)
        img = gaussian_blur(img, 9)
        img = highmask(img,0.5)
        img = canny_edge(img,250,350)
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
        img = gaussian_blur(img, 7)
        img = highmask(img, 0.5)
        img = canny_edge(img,450,510)
    elif imgclass == 4:
        img = contrastadd(img)
        img = highmask(img, 0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.5)
        img = canny_edge(img,500,510)
    else:
        img = contrastadd(img)
        img = highmask(img, 0.5)
        img = gaussian_blur(img, 5)
        img = highmask(img, 0.5)
        img = canny_edge(img,150,350)
    rgbimg = np.repeat(img[..., np.newaxis], 3, -1)
    return rgbimg
    

# Load training data
df_train = pd.read_csv("fungi_train.csv")
df_test = pd.read_csv("fungi_test.csv")

# Encode ClassId numerically
encoder = LabelEncoder()
df_train["ClassId"] = encoder.fit_transform(df_train["ClassId"])
num_classes = len(df_train["ClassId"].unique()) # should be 5 i believe

# train test splitting
train_list = []
val_list = []
for i in range(5):
    segment = df_train.iloc[i*1000 : (i+1)*1000]
    # scikitlearn train test split for segment
    train, val = train_test_split(segment, test_size=0.2)
    # convert train and val into lists and then extend train_list, val_list
    train_list.extend(train.values.tolist())
    val_list.extend(val.values.tolist())
train = pd.DataFrame(train_list, columns=df_train.columns)
val = pd.DataFrame(val_list, columns=df_train.columns)
# convert those paths to actual images
trainimgdata = []
trainclassdata = []
valimgdata = []
valclassdata = []
for _,row in tqdm(train.iterrows()):
    img = preprocess_image(row["Path"], row["ClassId"])
    trainimgdata.append(img)
    trainclassdata.append(int(row["ClassId"]))

for _,row in tqdm(val.iterrows()):
    img = preprocess_image(row["Path"], row["ClassId"])
    valimgdata.append(img)
    valclassdata.append(int(row["ClassId"]))

trainimgdata = np.array(trainimgdata)
trainclassdata = np.array(trainclassdata)
valimgdata = np.array(valimgdata)
valclassdata = np.array(valclassdata)

# One-hot encode the class labels
trainclassdata = to_categorical(trainclassdata, num_classes)
valclassdata = to_categorical(valclassdata, num_classes)

# Print shapes for debugging

print("making model")
# create the model  
base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
model = keras.Sequential([base_model,keras.layers.Flatten(),keras.layers.Dense(512, activation='relu'),keras.layers.BatchNormalization(),keras.layers.Dropout(0.3),keras.layers.Dense(num_classes, activation='softmax')])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5), loss='categorical_crossentropy',metrics=['accuracy'])
# Train and see train-test-split accuracy
print("done!")
print(f"trainimgdata shape: {trainimgdata.shape}")  # Should be (num_samples, height, width, channels)
print(f"trainclassdata shape: {trainclassdata.shape}")  # Should be (num_samples, num_classes)
model.fit(trainimgdata, trainclassdata, epochs=10)
print("model has been fit")
model.save("model.h5")
print("testing...")
# check the model accuracy on the testing data
test_loss, test_accuracy = model.evaluate(valimgdata, valclassdata)
print(f"Test accuracy: {test_accuracy:.4f}")

# test its accuracy on the test data
preds = []
for _, row in df_test.iterrows():
    # open row image
    imgpath = row["Path"]
    img = preprocess_image(imgpath, None)
    
    # Predict the class
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
    print(f"Predicted class for {imgpath}: {predicted_class}")
    preds.append(predicted_class)

# make a pandas array
df_submission = pd.DataFrame({"id": df_test["id"], "output": preds})
df_submission.to_csv("submission.csv", index=False)
print("Prediction complete. Results saved to submission.csv")

    
    
