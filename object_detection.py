import cv2

cap=cv2.VideoCapture("video/peoples.mp4")
ret,frame1=cap.read()
ret,frame2=cap.read()
while cap.isOpened():
    diff=cv2.absdiff(frame1,frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
    blur= cv2.GaussianBlur(gray,(5,5),0)
    _,th=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(th,None,iterations=3)
    contour,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame1,contour,-1,(0,255,0),2)
    total=len(contour)
    for i in contour:
        (x,y,w,h)=cv2.boundingRect(i)
        if cv2.contourArea(i)<700:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(frame1,f"Objects: {total}",(10,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    cv2.imshow("car",frame1)
    frame1=frame2
    ret,frame2=cap.read()

    if cv2.waitKey(60) & 0xFF ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()