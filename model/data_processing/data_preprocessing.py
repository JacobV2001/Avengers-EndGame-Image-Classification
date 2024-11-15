import os
import shutil
import cv2  # OpenCV library for image processing
from face_detection import get_cropped_image_if_2_eyes  # function to crop face image if two eyes are detected

# Define paths to datasets and folder for cropped images
path_to_data = "./dataset/"
path_to_cr_data = "./dataset/cropped/"

# Remove existing folder for cropped images and create a new one
shutil.rmtree(path_to_cr_data, ignore_errors=True)  # Removes folder if it exists
os.makedirs(path_to_cr_data, exist_ok=True)

# interate over folders in dataset folder (must only contain folders)
for img_dir in os.scandir(path_to_data):
    # set character & folder names
    character_name = img_dir.name
    if character_name == "cropped": continue # make sure a cropped folder doesnt appear inside cropped
    character_folder = os.path.join(path_to_cr_data, character_name)

    # create character folder 
    os.makedirs(character_folder, exist_ok=True)

    count = 1 # used for img name count

    # iterate through images in character file (must only contain img)
    for entry in os.scandir(img_dir.path):
        
        # function to get cropped image if two eyes are detected
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        
        # if cropping was successful
        if roi_color is not None:
            
            # set file name and path
            cropped_file_name = f"{character_name}{count}.png"
            cropped_file_path = os.path.join(character_folder, cropped_file_name)
            
            cv2.imwrite(cropped_file_path, roi_color) # save cropped image to specified path
            count += 1 # increase count for image name
