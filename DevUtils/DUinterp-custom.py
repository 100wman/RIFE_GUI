import cv2
import numpy as np
i0 = cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\0.png")
i1 = cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\1.png")
scale = 1.0 #开spilt后不建议调，不过可以复合用
h, w, c = i0.shape
i00 = i0[0::2, 0::2, :]
i01 = i0[0::2, 1::2, :]
i02 = i0[1::2, 0::2, :]
i03 = i0[1::2, 1::2, :]
black = np.zeros((h, w, c))
black[0::2, 0::2, :] = i00
cv2.imwrite(r"black-1.png", black)
black[0::2, 1::2, :] = i01
cv2.imwrite(r"black-2.png", black)
black[1::2, 0::2, :] = i02
cv2.imwrite(r"black-3.png", black)
black[1::2, 1::2, :] = i03
cv2.imwrite(r"black-4.png", black)
