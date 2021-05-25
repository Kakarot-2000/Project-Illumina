# Illumina

Illumina is a Web App built to perform real time emotion detection on live camera feed. I have implemented a Convolutional Neural Network and trained the network on fer2013 dataset hosted by Kaggle.
Detects upto 7 emotions for all identified faces in frame ['happy','neutral','fear','suprise','sad','disgust','anger'].
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
- Create a Virtual Environment and Install all dependencies from requirements.txt
```
  pip install -r requirements.txt
```
- Run app.py file
```
  python app.py
```
- Enter localhost address in browser. 

Note : Make sure there is good lighting in the room to get better accuracy.

Thats it ^_^

**Demo**
![sample1](sample1(1).gif)


