# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# define constants
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

def recognize_activity(model_path, classes_path, input_video=""):
    # Load the class labels
    classes = open(classes_path).read().strip().split("\n")
    
    # Initialize the frames queue to ensure only the most recent frames are used for activity recognition in this case its 16
    frames = deque(maxlen=SAMPLE_DURATION)
    
    # Load the human activity recognition model
    print("[INFO] Initializing the human activity recognition model...")

    net = cv2.dnn.readNet(model_path)
    
    # Grab a pointer to the input video stream
    print("[INFO] Establishing connection to the video stream...")

    cap = cv2.VideoCapture(input_video if input_video else 0)

    # Loop over frames from the video stream
    while True:
        # Read a frame from the video stream
        (success, frame) = cap.read()
        
        if not success:
            print("[INFO] No frames were retrieved from the video stream. Exiting...")
            break
        
        # Resize the frame and add to queue
        frame = imutils.resize(frame, width=400)
        frames.append(frame)
        
        if len(frames) < SAMPLE_DURATION:
            continue
        
        # Construct the blob and pass through the network
        blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        # Obtain predictions
        net.setInput(blob)
        outputs = net.forward()
        label = classes[np.argmax(outputs)]
        
        # Draw the predicted activity on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Human Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Specify the path to the trained model for human activity recognition.")
    ap.add_argument("-c", "--classes", required=True, help="Provide the path to the file containing class labels.")
    ap.add_argument("-i", "--input", type=str, default="", help="Optionally, specify the path to a video file. Leave empty for webcam input.")
    args = vars(ap.parse_args())
    
    # Call the activity recognition function
    recognize_activity(args["model"], args["classes"], args["input"])
