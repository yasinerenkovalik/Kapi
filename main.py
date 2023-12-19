import os
import pickle

import cv2
import face_recognition

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
imgBackground=cv2.imread("Resource/background.png")

folderModePath="Resource/Models"
modelPathList=os.listdir(folderModePath)
imgModeList=[]
for path in modelPathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

print("Loadng Encoded File")
file=open("EncodeFile.p","rb")
encodeListKnownWithIds=pickle.load(file)
file.close()

encodeListKnow,studentIds=encodeListKnownWithIds

#print(studentIds)
print(" Encoded File Loadng ")
while True:
    success,img=cap.read()


    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)

    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)

    cv2.imshow("Wabcam",img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)