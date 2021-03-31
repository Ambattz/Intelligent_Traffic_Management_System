import cv2 
camera_port = 0
camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
return_value, image = camera.read()
cv2.imwrite("image.png", image)

camera.release()
cv2.destroyAllWindows()