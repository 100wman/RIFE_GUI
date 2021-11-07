import cv2
import glob

inputs = glob.glob("*.png")
png_i = 0
while True:
    img = cv2.imread(inputs[png_i])
    png_i = (png_i + 1) % len(inputs)
    cv2.imshow('image', img)
    t = cv2.waitKey(10)
    if t != -1:
        cv2.destroyAllWindows()
        break
