import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('Original_Reference.jpg')
    std = cv2.imread('Target_Img.png')
    img_hist, img_cumhist = histGet(img)
    std_hist, std_cumhist = histGet(std)
    dst = histSpec(img, std, img_cumhist, std_cumhist)
    dst_hist, dst_cumhist = histGet(dst)
    histPlotShow(img, std, dst, img_hist, std_hist, dst_hist)

def histGet(src):
    row = src.shape[0]
    col = src.shape[1]

    hist = np.zeros(256, dtype=np.float32)
    for i in range(row):
        for j in range(col):
            index = src[i, j]
            hist[index] += 1

    cumHist = np.zeros(256, dtype=np.float32)
    cumHist[0] = hist[0]
    for i in range(1, 256):
        cumHist[i] = cumHist[i - 1] + hist[i]

    return hist, cumHist

def histSpec(img, std, img_cumhist, std_cumhist):
    img_pixelnum = img.shape[0] * img.shape[1]
    std_pixelnum = std.shape[0] * std.shape[1]
    lut = np.zeros(256, dtype=np.float32)

    for i in range(256):
        absList = [abs(img_cumhist[i] /img_pixelnum - std_cumhist[j] /std_pixelnum) for j in range(256)]
        idx = absList.index(min(absList))
        lut[i] = idx

    lut = np.uint8(lut + 0.5)
    dst = cv2.LUT(img, lut)
    return dst

def histPlotShow(img, std, dst, img_hist, std_hist, dst_hist):
    DN = np.arange(256) - 0.5
    img_texts = ['original image', 'target image', 'output image']
    bar_texts = ['original', 'target', 'output']

    fig = plt.figure(figsize=(12, 8), dpi=100)

    for i,imgs in enumerate([img, std, dst]):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgs)
        plt.xticks([])
        plt.yticks([])
        plt.title(img_texts[i])

    for i,hists in enumerate([img_hist, std_hist, dst_hist]):
        plt.subplot(2, 3, i + 4)
        plt.bar(DN, hists, color='b', width=1)
        plt.xlim([-0.5, 255.5])
        plt.title(bar_texts[i])

    #plt.savefig('output_result.png')
    plt.show()

if __name__=='__main__':
    main()
