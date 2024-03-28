import cv2
import face_recognition

# Load the passport image
passport_image = face_recognition.load_image_file("teudat_zehut.jpg")

# Load your image
my_image = face_recognition.load_image_file("my_pic.jpg")

# Find face encodings for the passport image
passport_face_encodings = face_recognition.face_encodings(passport_image)

# Find face encodings for your image
my_face_encodings = face_recognition.face_encodings(my_image)

# Check if there are faces in both images
if len(passport_face_encodings) > 0 and len(my_face_encodings) > 0:
    # Compare the faces
    match = face_recognition.compare_faces(passport_face_encodings, my_face_encodings[0])

    if match[0]:
        print("It's likely the same person in both images.")

    else:
        print("Faces don't match.")

else:
    print("No faces found in one or both images.")
