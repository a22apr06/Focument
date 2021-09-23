import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog


img_counter = 0
def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

ROOT = tk.Tk()
ROOT.withdraw()
location = simpledialog.askstring(title="Location",
                                  prompt="Do you want to use the default image? If yes, press Y else to take photo press N.")#Count is just used to save image names with index 

if location.upper() == 'Y':
    image = cv2.imread('test3.jpg')
    img_name = "opencv_frame_{}.png".format(img_counter)

else:
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")


    while True:
        ret, frame = cam.read()
        if not ret:
            print("")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 32:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            image = cv2.imread("opencv_frame_{}.png".format(img_counter))
            cam.release()
            cv2.destroyAllWindows()
        elif k%256 == 27:
            print("no problem")
            cv2.destroyAllWindows()
            cam.release()


orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 0, 50)
orig_edged = edged.copy()

(contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)


for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break


approx = rectify(target)
pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

M = cv2.getPerspectiveTransform(approx,pts2)
dst = cv2.warpPerspective(orig,M,(800,800))

cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("Scanned Image", th4)
cv2.imshow("Orignal image",image)
cv2.imwrite(f'{img_name}',th4)

cv2.waitKey(0)
cv2.destroyAllWindows()