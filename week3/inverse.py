import cv2

def inverseImg(src):
    dst = 255 - src
    return dst

img = cv2.imread(r'coin.jpg', 0)
cv2.imshow('input', img)

img2 = inverseImg(img)
cv2.imshow('output',img2)
cv2.waitKey()

cv2.destroyAllWindows() #注意不要写成destory
#cv2.imwrite(r'coin_out.jpg', img2)