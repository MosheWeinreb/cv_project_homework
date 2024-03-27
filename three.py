import cv2
import face_recognition

reference_image = face_recognition.load_image_file("face and id.jpg")
id_image = face_recognition.load_image_file("face.jpg")
reference_face_locations = face_recognition.face_locations(reference_image)
id_face_locations = face_recognition.face_locations(id_image)

reference_face_encodings = face_recognition.face_encodings(reference_image, reference_face_locations)
id_face_encodings = face_recognition.face_encodings(id_image, id_face_locations)

print(id_face_encodings)
for id_face_encoding in id_face_encodings:
    matches = face_recognition.compare_faces(reference_face_encodings, id_face_encoding)


    if True in matches:
        print("Verification successful: The ID photo matches the reference photo.")
    else:
        print("Verification failed: The ID photo does not match the reference photo.")
