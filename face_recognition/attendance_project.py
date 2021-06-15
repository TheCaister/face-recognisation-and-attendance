import cv2
import numpy as np
import face_recognition
import os

path = "images_attendance"
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)

# Instead of manually initialising every image, we can run a for loop
for cl in my_list:
    # Read current image, append it to the images list and append its name to the class_names list
    current_image = cv2.imread(f'{path}/{cl}')
    images.append(current_image)
    class_names.append(os.path.splitext(cl)[0])
print(class_names)


# Function for finding the encodings of a list of images
def find_encodings(images):
    encode_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list


encode_list_known = find_encodings(images)
print("Encoding Complete")

# Initialising the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # Reducing the size of the frames for speed and converting them to RGB
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # The webcam could capture multiple faces
    # So get the location of all these faces and pass them in face_encodings
    faces_current_frame = face_recognition.face_locations(img_small)
    encodings_current_frame = face_recognition.face_encodings(img_small, faces_current_frame)

    # Iterating through every encoding and location
    for encode_face, face_location in zip(encodings_current_frame, faces_current_frame):
        # See if there are any matches
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        # Return the distances
        face_distance = face_recognition.face_distance(encode_list_known, encode_face)
