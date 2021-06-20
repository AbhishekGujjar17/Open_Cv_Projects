import cv2

from random import randrange


#loading a pre-trained data

trained_face_data = cv2.CascadeClassifier('open_cv/haarcascade_frontalface_default.xml')


#choose an image to detect face
#img = cv2.imread(r"open_cv/abhishekprofile.jpg")  #for one image

#real time web-camera/video

webcam = cv2.VideoCapture(0) #for video-file, instead of 0 place the videofile location

while True:

        successful_frame_read, frame = webcam.read()

        #convert into grayscale
        grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #detect_face
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

        #draw rectangle around face
        for (x,y,w,h) in face_coordinates:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)
            
            
            
        cv2.imshow('Abhishek program to detect face', frame)
        key = cv2.waitKey(1)

        if key == 81 or key == 113:  #if q is pressed
            break


webcam.release()       


print("code completed")