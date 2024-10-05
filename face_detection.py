import cv2

case=cv2.CascadeClassifier("face.xml")
#img=cv2.imread("images/ronaldo.jpg")
cap=cv2.VideoCapture("walk.mp4")
while cap.isOpened():
    ref,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    face=case.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("img",img)
    if cv2.waitKey(1) ==ord("q"):
       break

cap.release()
cv2.destroyAllWindows()