import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def Process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

while True:
    ref, frame = cap.read()
    if not ref:
        print("Failed to grab frame")
        break

    # Resize before processing
    frame_resized = cv2.resize(frame, (32, 32))

    img = Process(frame_resized)

    #cv2.imshow("Processed Frame", frame)
    img = img.reshape(1, 32, 32, 1)
    #prediction
    classes=np.argmax(img)
    predictions=model.predict(img)
    prob=np.amax(predictions)
    print(predictions,prob)
    if prob>0.65:
        cv2.putText(frame,str(classes) +" "+str(prob),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("image",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
