import cv2
import face_recognition
import pickle
import os

folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds=[]

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #print(path)
    #print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])
print(studentIds)



def findEncoding(imagesList):
    encodeList=[]
    for img in imagesList:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print("encoding strated")
encodeListKnow=findEncoding(imgList)
print(encodeListKnow)
encodeListKnowWithId=[encodeListKnow,studentIds]
print("encoding complete")

file=open("EncodeFile.p","wb")
pickle.dump(encodeListKnowWithId,file)
file.close()
print("file saved")



