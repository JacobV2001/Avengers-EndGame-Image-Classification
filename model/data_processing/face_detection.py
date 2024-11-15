import cv2

# load pre-trained haar cascade classifiers
face_cascade = cv2.CascadeClassifier('./opencv/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

# function to get cropped image of the face
def get_cropped_image_if_2_eyes(image_path):

    # read the image given the path
    img = cv2.imread(image_path)

    # check if image was loaded
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # convert the image to grayscale (needed for face and eye detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the gray scale with (haar cascade classifier)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # if no faces are detected, break
    if len(faces) == 0:
        print("No faces detected.")
        return None

    # loop over each face (if multiple faces detected)
    for (x, y, w, h) in faces:
        """
        There are features that appear in color and
        other features that appear in grayscale
        """
        # define regions of interest for face in gray scale and color
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        # detect eyes within the region of interes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # two eyes needed for more accurate results
        if len(eyes) >= 2:
            return roi_color # return color version of image
    
    # appears here if face detected but less than 2 eyes
    print("Less than two eyes detected.")
    return None
