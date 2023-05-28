import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "ImageAttendence"
images = []
Names = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    Names.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markattendence(name):
    with open("Attendence.csv", "r+") as f:
        mydata = f.readlines()
        namelist = []
        for lines in mydata:
            entry = lines.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')


encodelistknown = findEncodings(images)
print("Encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurrFrame = face_recognition.face_locations(imgS)
    encodingsCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    for encodeface, faceloc in zip(encodingsCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedis = face_recognition.face_distance(encodelistknown, encodeface)
        print(facedis)
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = Names[matchindex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.66, (255, 255, 255), 2)
            markattendence(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)