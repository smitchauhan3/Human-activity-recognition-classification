# Human-activity-recognition-classification
Live human activity detection refers to the process of identifying and monitoring human movements in real-time. The most common approach involves two steps: feature extraction and classification. The aim of this project is to provide an overview of the current state-of-the-art techniques for live human activity detection. 

Important Things About This Project:
•	This is a laptop or desktop application.
•	In this project we have used LSTM algorithm to develop this project.
•	In this project we collects the numpy array to trace the activity and create dataset.
•	Application compare the activity detection keypoints to the dataset numpy array and whichever is the most relevant  activity, it will appear on the screen.
•	We are using mediapipe holistic to extract keypoints from body. We used keras and tensorflow to create LSTM model.


Approach To The Problem:
1.	Extract holistic keypoints
2.	Collect keypoints and values for training and testing
3.	Build and train LSTM neural network
4.	Make predictions
5.	Evaluation using confusion matrix and accuracy


----------------------------------------------------------------------------------
  How to Execute Program
----------------------------------------------------------------------------------
1. First of all u have to run 'record_actions.py' for recording of your activity in form of numpy data.
2. After creating numpy data of your Activity you have to run 'make_model.py' for make a LSTM model. you can alse use our pretrained model 
but I suggest you have to record Activity yourself for better results.
3. 'main.py' is a main file of real time Activity Recognization and Classification.
