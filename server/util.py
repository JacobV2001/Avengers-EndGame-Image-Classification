import joblib
import json
import numpy as np
import base64 # to decode base64 images
import cv2
from wavelet import w2d

__class_name_to_number = {} # mapping class names to numbers
__class_number_to_name = {} # mapping numbers to class names

__model = None # for trained model

def classify_image(image_base64_data, file_path=None):
    """
    This function takes an image, process it, and returns the classification result
    """

    # crop image to faces with two eyes
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = [] # list to hold classification results
    for img in imgs:
        # resize image
        scalled_raw_img = cv2.resize(img, (32, 32))
        # perform wavele transformation
        img_har = w2d(img, 'db1', 5)
        # resize transformed image
        scalled_img_har = cv2.resize(img_har, (32, 32))
        # stack raw and transformed image
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        # length of final image array for prediction
        len_image_array = 32*32*3 + 32*32

        # reshape image to flat array and convert to float
        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]), # get predicted class
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0], # get class probability
            'class_dictionary': __class_name_to_number # return class dictionary
        })

    # return classification results
    return result

def class_number_to_name(class_num):
    """
    Convert class number to class name
    """
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    """
    Load the saved model adn class dictionary
    """
    print("loading saved artifacts...start")

    with open("server/artifacts/class_dictionary.json", "r") as f:
        global __class_name_to_number
        global __class_number_to_name
        __class_name_to_number = json.load(f) # load class dictionary from json
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()} # map reversed dictionary 

    global __model
    if __model is None: # if model not loaded yet
        with open('server/artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f) # load the trained model
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    convert a base64 string to OpenCV
    '''

    # extract the actual image dataset from base64
    encoded_data = b64str.split(',')[1]
    # decode base64 into numpy array
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    # convert array into OpenCV image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    detech faces with 2 eyes in them
    """
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path) # load image from file if provided
    else:
        img = get_cv2_image_from_base64_string(image_base64_data) # decode base64 image if image not provided

    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces in image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = [] # list to hold cropped faces if multiple
    for (x,y,w,h) in faces:
            
            # extract region of interest
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # detect eyes within ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # if 2 eyes detected, consider face for classification
            if len(eyes) >= 2: cropped_faces.append(roi_color)

    return cropped_faces # return list of cropped faces

if __name__ == '__main__':
    load_saved_artifacts() # load model and class dictionary
