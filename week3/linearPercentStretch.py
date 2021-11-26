import cv2
import numpy as np

def linearPercentStretch(src, percent):
    src = np.float64(src)
    cut = np.floor(256 * percent + 0.5) # 计算需要裁减的数量 floor() 向下取整
    minvalue = np.min(src)
    maxvalue = np.max(src)

    newminvalue = minvalue + cut
    newmaxvalue = maxvalue - cut

    #row, col = src.shape
    #ans = np.array((row, col), dtype = np.float64)
    ans = 255.0 * (src - newminvalue) / (newmaxvalue + newminvalue)
    ans = ans + 0.5
    ans[ans > 255] = 255
    ans[ans < 0] = 0
    ans = np.uint8(ans)
    return ans

img = cv2.imread(r'Einstein.tif', 0)
cv2.imshow('input', img)

img2 = linearPercentStretch(img, 0.02)
cv2.imshow('output', img2)

#cv2.imwrite(r'Einstein_out.tif', img2)
cv2.waitKey()
cv2.destroyAllWindows()