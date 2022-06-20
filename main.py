import cv2
import numpy as np
import face_recognition

imgREWIKADAMAYANTISIMBOLON191401127 = face_recognition.load_image_file('ImageBasic/REWIKA DAMAYANTI SIMBOLON-191401127.jpg')
imgREWIKADAMAYANTISIMBOLON191401127 = cv2.cvtColor(imgREWIKADAMAYANTISIMBOLON191401127, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/REWIKA DAMAYANTI SIMBOLON-191401127test.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgREWIKADAMAYANTISIMBOLON191401127)[0]
encodeREWIKADAMAYANTISIMBOLON191401127 = face_recognition.face_encodings(imgREWIKADAMAYANTISIMBOLON191401127)[0]
cv2.rectangle(imgREWIKADAMAYANTISIMBOLON191401127, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeREWIKADAMAYANTISIMBOLON191401127], encodeTest)
faceDis = face_recognition.face_distance([encodeREWIKADAMAYANTISIMBOLON191401127], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('REWIKADAMAYANTISIMBOLON191401127', imgREWIKADAMAYANTISIMBOLON191401127)
cv2.imshow('REWIKADAMAYANTISIMBOLON191401127test', imgTest)
cv2.waitKey(0)