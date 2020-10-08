# Real_Time_Emotion_Detection

Here, to perform emotion detection on live camera feed I have implemented a Convolutional Neural Network and trained the network on fer2013 dataset hosted by Kaggle.
Achieved 65%(top 10 percentile accuracy) on validation dataset .
Then the saved model was used to perform emotion detection from webcamera input using OpenCV.
Dataset Downloaded from : https://www.kaggle.com/deadskull7/fer2013

Steps to Run:
1. Clone the opencv repo ('https://github.com/opencv/opencv')
2. Open predict.py file and add the path file to the 'opencv-master\data\haarcascades\haarcascade_frontalface_default.xml' file in this line 
   face_haar_cascade = cv2.CascadeClassifier('C:\add_path_to_opencv_here\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')  
3. Run predict.py file in any python ide (press q to exit) . 
Thats it ^_^
