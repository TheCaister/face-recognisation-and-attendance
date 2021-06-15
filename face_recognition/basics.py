import cv2
import numpy as np
import face_recognition

# Loading the image and converting it to RGB
img_cena = face_recognition.load_image_file("images_basic/john_cena.jpg")
img_cena = cv2.cvtColor(img_cena, cv2.COLOR_BGR2RGB)
img_cena_test = face_recognition.load_image_file("images_basic/john_cena_test.jpg")
img_cena_test = cv2.cvtColor(img_cena_test, cv2.COLOR_BGR2RGB)

# Get the first face in the image
# Specifically, get the coordinates for the bounding box
face_location = face_recognition.face_locations(img_cena)[0]
encode_cena = face_recognition.face_encodings(img_cena)[0]
# Drawing a rectangle around the detected face
cv2.rectangle(img_cena, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (255, 0, 255), 2)

# Detecting face in second image
face_location_test = face_recognition.face_locations(img_cena_test)[0]
encode_cena_test = face_recognition.face_encodings(img_cena_test)[0]
cv2.rectangle(img_cena_test, (face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (255, 0, 255), 2)

# Comparing faces to see if the faces in the 2 images are similar
results = face_recognition.compare_faces([encode_cena], encode_cena_test)
print(results)

cv2.imshow("John Cena", img_cena)
cv2.imshow("John Cena Test", img_cena_test)
cv2.waitKey(0)