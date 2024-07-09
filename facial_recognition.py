import cv2
import os
from deepface import DeepFace
import pandas as pd

# Function to create faces folder if it doesn't exist
def create_faces_folder():
    if not os.path.exists("faces"):
        os.makedirs("faces")


def register_face():
    # Prompt for password
    password = input("Enter password to register a face: ")

    # Check if password is correct
    if password != "admin":
        print("Incorrect password. Access denied.")
        return

    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Start video capture
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot capture video.")
            break

        # Display the resulting frame
        cv2.imshow('Register Face', frame)

        # Check for space bar press to capture the face
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Ask user for name for the registered face
                name = input("Enter a name for the registered face: ")
                name = name.strip()
                if not name:
                    print("Name cannot be empty. Please enter a valid name.")
                else:
                    # Save the detected face image with the provided name
                    face_image = os.path.join("faces", f"{name}.jpg")
                    cv2.imwrite(face_image, frame[y:y + h, x:x + w])

                    print(f"Face registered with name: {name}")

            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()


def recognize_face(img):
    faces_folder = "faces"
    if not os.listdir(faces_folder):
        print("No registered faces found.")
        return False  # Return False if there are no registered faces

    # Recognize faces with cosine similarity metric and threshold set to 0.3
    recognized_faces = DeepFace.find(img, db_path=faces_folder, enforce_detection=False, distance_metric='cosine',
                                     threshold=0.3)

    # Check if the recognition result is an empty DataFrame
    if any(isinstance(result, pd.DataFrame) and result.empty for result in recognized_faces):
        print("No face recognized.")
        return False

    # Check if any faces are recognized
    if recognized_faces[0].empty:
        return False
    else:
        # Get the name of the registered face from the DataFrame
        registered_name = recognized_faces[0]['identity']
        print("Face recognized:", registered_name)
        return registered_name


# Function to display access granted or denied
def display_access(frame, status, name="Unknown"):
    if status:
        cv2.putText(frame, f"Access Granted: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Access Denied: Unknown face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def scan_faces():
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Start video capture
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot capture video.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Recognize faces
            recognized_names = recognize_face(frame[y:y + h, x:x + w])

            # Debug print
            print("Recognition result:", recognized_names)

            # Check if any faces are recognized
            if recognized_names is not False:
                # Display access granted with the recognized name
                recognized_names = [os.path.splitext(os.path.basename(name))[0] for name in recognized_names]

                display_access(frame, True, name=recognized_names)
            else:
                # Display access denied
                display_access(frame, False)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()




# Main program
if __name__ == "__main__":
    create_faces_folder()  # Create 'faces' folder if it doesn't exist
    while True:
        print("\nChoose an option:")
        print("1. Scan a face")
        print("2. Register a face")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            scan_faces()
        elif choice == '2':
            register_face()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please choose again.")
