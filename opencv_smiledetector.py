import cv2

trained_face_data = cv2.CascadeClassifier(
    'open_cv/haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('open_cv/haarcascade_smile.xml')


webcam = cv2.VideoCapture(0)

while True:

   (succesful_read, frame) = webcam.read()

   if not succesful_read:
       break

   grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

   for (x, y, w, h) in face_coordinates:

       cv2.rectangle(frame, (x, y), (x+w,y+h), (255, 0, 0), 2)
       the_face = frame[y:y+h, x:x+w]
       the_face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
       smile_coordinates = trained_smile_data.detectMultiScale(
           the_face_gray, scaleFactor=1.7, minNeighbors=20)

       if len(smile_coordinates) > 0:
           cv2.putText(frame, 'smiling', (x-70, y+317), fontScale=3,
                       fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255, 255, 255))

   cv2.imshow('Smile Detector',frame)
   cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()

print('Code Completed')






