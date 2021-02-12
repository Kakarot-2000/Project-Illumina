# Real_Time_Emotion_Detection

Here, to perform emotion detection on live camera feed I have implemented a Convolutional Neural Network and trained the network on fer2013 dataset hosted by Kaggle.
Achieved 65%(top 10 percentile accuracy) on validation dataset .
Then the saved model was used to perform emotion detection from webcamera input using OpenCV.
Dataset Downloaded from : [link](https://www.kaggle.com/deadskull7/fer2013)

**Directory Structure:**
- templates : folder containing html files
- app.py : py file containing the server side code
- emotion_detection.py : neural network implementation
- predict.py : py file to run detection in a window instead of browser
- fer.h5 : saved model architecture
- fer.json : saved model weights

**Steps to Run:**
- Run app.py file in any python ide
- Enter localhost address in browser

Thats it ^_^
