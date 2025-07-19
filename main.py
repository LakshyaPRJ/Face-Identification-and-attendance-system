import face_recognition
import cv2
import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#Load known faces
lakshya_image = face_recognition.load_image_file("faces/lakshya.jpg")
lakshya_encoding = face_recognition.face_encodings(lakshya_image)[0]

anshita_image = face_recognition.load_image_file("faces/anshita.jpg")
anshita_encoding = face_recognition.face_encodings(anshita_image)[0]

kalpana_image = face_recognition.load_image_file("faces/kalpana.jpg")
kalpana_encoding = face_recognition.face_encodings(kalpana_image)[0]

kunal_image = face_recognition.load_image_file("faces/kunal.jpg")
kunal_encoding = face_recognition.face_encodings(kunal_image)[0]

lakhan_image = face_recognition.load_image_file("faces/lakhan.jpg")
lakhan_encoding = face_recognition.face_encodings(lakhan_image)[0]

siddhi_image = face_recognition.load_image_file("faces/siddhi.jpg")
siddhi_encoding = face_recognition.face_encodings(siddhi_image)[0]

known_face_encodings = [lakshya_encoding, kalpana_encoding, anshita_encoding, lakhan_encoding, kunal_encoding, siddhi_encoding]
known_face_names = ["Lakshya", "Kalpana", "Anshita", "Lakhan", "Kunal", "Siddhi"]

#list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

#get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline = "")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]

        #Add text if person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendence", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
