import cv2
import numpy as np

def connect(img, y, x, low):
    row, col = img.shape[0 : 2]
    neighbour = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    for k in range(8):
        yy = y + neighbour[k, 0]
        xx = x + neighbour[k, 1]
        if yy >= 0 and yy < row and xx >= 0 and xx < col:
            	if img[yy, xx] >= low and img[yy, xx] != 255:
                	img[yy, xx] = 255
                	connect(img, yy, xx, low)


def LOG(src):
    #make border
    # dst = np.zeros(src.shape, dtype=np.float64)
    src = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    dst = np.zeros(src.shape, dtype=np.float64)
    src = np.float64(src)

    #define template
    template = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]], dtype = np.float64)
    row, col = template.shape
    radius = int(row/2)

    #filter
    srcRow,srcCol = src.shape
    for i in range(2, srcRow-2):
        for j in range(2, srcCol-2):
            # dst[i, j] = np.sum(src[i:i+5,j:j+5]*template)
            temp = src[i-2:i+3,j-2:j+3]
            dst[i, j] = np.sum(temp*template)

    #post processing
    dst = np.abs(dst)
    dmin = np.min(dst)
    dmax = np.max(dst)
    dst = 255.0 / (dmax - dmin)*(dst - dmin)
    dst = dst[0 + radius: srcRow - radius, 0 + radius: srcCol - radius]
    dst = np.uint8(dst + 0.5)

    dstRow, dstCol = dst.shape
    res = dst.copy()
    up = 40
    low = 38
    for i in range(dstRow):
        for j in range(dstCol):
            if res[i,j]>up:
                res[i,j] = 255
                connect(res,i,j,low)
    res = np.uint8(res)
    result = np.zeros(res.shape, dtype = np.uint8)
    result[res==255] = 255
    return result

img = cv2.imread('Lena.tif', 0)
dst = LOG(img)

cv2.imshow('LOG_Lena', dst)
cv2.waitKey()
cv2.destroyAllWindows()