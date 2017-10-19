
import cv2
import numpy as np

room_img = np.ones(shape=(1500, 3000), dtype=np.uint8) * 255

room_img = cv2.rectangle(room_img, (0, 0), (3000, 1500), 0, 5)

cv2.imwrite('resources/maps/generated.png', room_img)

# cv2.imshow('1', room_img)
# cv2.waitKey(0)
