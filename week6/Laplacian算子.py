import cv2
import numpy as np

def main():
    origin = cv2.imread('homework_img.tif', 0)
    backgroundFalse = LaplacianFilter(origin, False)
    backgroundTrue = LaplacianFilter(origin, True)
    win_sets = ['original img', 'laplacian filter(no background)', 'laplacian filter(background)']
    img_sets = [origin, backgroundFalse, backgroundTrue]
    for i, img in enumerate(win_sets):
        cv2.namedWindow(win_sets[i], 0)
        cv2.imshow(win_sets[i], img_sets[i])
    cv2.waitKey(0)
    cv2.imwrite('gray_original.tif', origin)
    cv2.imwrite('laplacian filter(no background).tif', backgroundFalse)
    cv2.imwrite('laplacian filter(background).tif', backgroundTrue)

def LaplacianFilter(src, background = False):
    #make border
    src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    src = np.float64(src)

    if background == True:
        template = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)

    if background == False:
        template = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    #compute
    g = np.zeros(src.shape, dtype=np.float64)
    rows, cols = src.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
                temp = src[i- 1: i + 2, j - 1: j + 2]
                g[i, j] = np.sum(temp * template)

    #clip border
    g = g[1: rows-1, 1: cols-1]
    g = np.abs(g)

    #strench
    if background == False:
        gmin = np.min(g)
        gmax = np.max(g)
        g = 255.0 / (gmax - gmin) * (g - gmin)
        g = np.uint8(g + 0.5)
        # _, g = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)

    if background == True:
        gmin = np.min(g)
        gmax = np.max(g)
        g = 255.0 / (gmax - gmin) * (g - gmin)
        g = np.uint8(g + 0.5)

    return g

if __name__ == '__main__':
    main()