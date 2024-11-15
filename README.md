# Endgame Image Classifier

## Overview

The **Endgame Image Classifier** is a machine learning project that classifies images of people based on their facial features and compares them to characters from the movie *Avengers: Endgame*. The goal is to determine which character an individual resembles the most using facial recognition and image classification techniques.

This project serves as both a fun lookalike tool and a learning platform for image classification and machine learning.It demonstrates key steps in creating an image classification pipeline, including data gathering, image processing, model training, and building a simple web interface for interaction

## Features

- **Lookalike Classification**: Upload an image, and the model predicts which *Avengers: Endgame* character you resemble the most.
- **SVM Model**: Uses Support Vector Machines (SVM) for facial feature-based image classification.
- **Facial Feature Extraction**: Utilizes OpenCV and Haar Cascade classifiers to extract key facial features.
- **Local Web Interface**: Built with HTML, CSS, and JavaScript for user interaction.
- **Web Scrapping Extraction**: Uses Selenlium to automade downloading of endgame character images.


Required Python libraries:
- `scikit-learn` (for machine learning models)
- `opencv-python` (for image processing)
- `numpy` (for numerical operations)
- `matplotlib` (for plotting and visualization)
- `selenium` (for image scraping)
- `flask` (for the local web server)

You can install the necessary Python libraries by running:
pip install -r requirements.txt


## Dataset and Image Scraping

The dataset used to train the classifier was gathered by scraping images of *Avengers: Endgame* characters from Google Chrome Images. The image scraping process was automated using the `selenium` module.

### Image Scraping:
- The script searches for a list of *Avengers: Endgame* characters on Google Images.
- Using `selenium`, it automates the browser to load the image results and download images of the characters.
**Note: The image scraping process was performed for educational and research purpose only.**

### Image Preprocessing:
- Once images were downloaded, they were cropped to focus on the faces using facial detection with OpenCV’s Haar Cascade Classifiers.

### Dataset Creation:
- After scraping and preprocessing, the images were labeled according to the character they represented.
- The dataset was then split into training and testing sets (80/20 split).

## How It Works

### Image Preprocessing:
- The first step is to process the uploaded image to detect any faces with two eyes using OpenCV's Haar Cascade Classifier.
- Once a face is detected, the image is cropped to focus solely on the facial features (eyes, nose, mouth) for better recognition accuracy.
  
### Feature Extraction:
- Facial landmarks are extracted from the cropped images using OpenCV.
- The extracted features (e.g., shading between eyes and other facial details) are used to create a feature vector that represents each face.

### Model Selection and Training:
- Grid search was used to tune the hyperparameters and find the best configuration for the model.
- The **Support Vector Machine (SVM)** algorithm was selected based on performance with respect to the dataset.
- The dataset was split into **80% training** and **20% testing** data to evaluate the model's performance.

### Classification:
- After training, the SVM model is used to classify the uploaded image and predict which *Avengers: Endgame* character it most closely resembles.
- The model outputs the name of the character with the highest classification probability.


## Performance

The model achieved an accuracy of **89%** on the test dataset.
SVM Test Accuracy: 0.8923076923076924

### Classification Report for SVM:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.67   | 0.80     | 3       |
| 1     | 1.00      | 1.00   | 1.00     | 4       |
| 2     | 1.00      | 0.73   | 0.84     | 11      |
| 3     | 1.00      | 1.00   | 1.00     | 5       |
| 4     | 0.89      | 0.73   | 0.80     | 11      |
| 5     | 0.91      | 1.00   | 0.95     | 20      |
| 6     | 0.80      | 1.00   | 0.89     | 4       |
| 7     | 0.70      | 1.00   | 0.82     | 7       |

**Accuracy**: 0.89  
**Macro avg**: 0.91 precision, 0.89 recall, 0.89 f1-score  
**Weighted avg**: 0.91 precision, 0.89 recall, 0.89 f1-score


## How to Use

### Step 1: Clone the Repository

Clone the repository to your local machine:
`git clone https://github.com/your-username/endgame-image-classifier.git
cd endgame-image-classifier


### Step 2: Install Dependencies

Install the required libraries by running:
`install -r requirements.txt



### Step 3: Start the Local Server

To run the application locally, use the following command:
`python server/server.py


This will start the web server, and you can access the application by opening the app.html file in your browser.

### Step 4: Upload Your Image

1. Go to the web interface in your browser.
2. Upload an image with a clear face.
3. The classifier will predict which *Avengers: Endgame* character you look like based on facial features.



## Notes

- **Accuracy**: The model performs well with an accuracy of around 89%. To improve, you could consider using a larger dataset or trying different machine learning algorithms.
- **Web Deployment**: This application is not deployed online but can be hosted on platforms like **Heroku** or **AWS** if desired.


## Credits

- **Face Detection**: OpenCV's Haar Cascade Classifiers were used for detecting and processing faces.
- **Avengers Characters**: The dataset contains publicly available images of characters from *Avengers: Endgame*, scraped for educational purposes.

