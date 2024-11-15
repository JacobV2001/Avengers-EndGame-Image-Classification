import numpy as np
import cv2
import os
from wavelet_transform import w2d

def prepare_data():
    # oath where images are stored
    data_dir = 'dataset/cropped'

    X = [] # feature vector list
    y = [] # labels
    class_dict = {} # dictionary to map characters

    # iterate through character subfolder
    for idx, character_name in enumerate(os.listdir(data_dir)):
        
        # get name path for character folder 
        character_folder = os.path.join(data_dir, character_name)

        if not os.path.isdir(character_folder): # skip non-folders
            continue
        
        # add character to dictionary
        class_dict[character_name] = idx

        # iterate through images in character list
        for image_file in os.listdir(character_folder):

            # get path and read image using OpenCV
            image_path = os.path.join(character_folder, image_file)
            img = cv2.imread(image_path)

            # skip if image cannot be used
            if img is None:
                print(f"Skipping image {image_path}, unable to read.")
                continue

            # resize image to 32x32 pixels (good size & to keep image size consistency)
            scalled_raw_img = cv2.resize(img, (32, 32))

            # apply waveley transformation (using db1 with 5 decomposition levels)
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32)) # resize image to same size as scalled_raw_image for consistency

            # cobine raw image and wavelet transformation image into a single feature vector
            combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
            X.append(combined_img) # add to feature list
            y.append(idx) # add label

    # convert lists to numpy arrays and reshape the model
    X = np.array(X).reshape(len(X), 4096).astype(float)
    return X, y, class_dict
