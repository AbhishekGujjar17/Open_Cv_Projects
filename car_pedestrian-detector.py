import cv2

#applying cascade classifier on a pre tained data
trained_pedestrian_data = cv2.CascadeClassifier('open_cv/haarcascade_fullbody.xml')
trained_car_data = cv2.CascadeClassifier('open_cv/cars_detector.xml')


video = cv2.VideoCapture('open_cv/video.avi')

while True:

    (successful_frame_read,frame) = video.read()

    if successful_frame_read:
        grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(grayscaled_img)
    car_coordinates = trained_car_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in pedestrian_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('Abhishek program to detect Car and Pedestrians',frame)

    key = cv2.waitKey(1)

    if key == 81 or key ==113:
        break


video.release()
print("code completed")
        


