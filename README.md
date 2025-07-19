Face Recognition Attendance System:
This project is a real-time face recognition-based attendance system built using Python, OpenCV, and the face_recognition library. It captures video from the webcam, identifies known faces using facial embeddings, and logs attendance automatically into a date-wise CSV file.
With a simple setup of labeled images, it can accurately detect and mark present individuals, making it ideal for classrooms, small offices, or personal experiments with computer vision.

Key Features:
* Live face detection via webcam
* Face recognition using deep learning-based encodings
* Automatic attendance logging in a .csv file with timestamp
* One-time attendance entry per person per session
* Easily extendable: just add new images to the faces/ folder

Tech Stack:
* Python
* OpenCV
* face_recognition (dlib-based)
* NumPy
* CSV module

Output:
* A live webcam window shows recognized faces with a “Present” label.
* A CSV file named like 2025-07-19.csv is created with rows like:
  Lakshya,14-22-05
  Anshita,14-23-12

--> To stop, press "Q" on keyboard.
