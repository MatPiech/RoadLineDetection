import cv2
import numpy as np
from matplotlib import pyplot as plt

ImageList = []
rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]

#Function to show pictures from list
def showPictures(ImageList, cmap='gray'):
    length = len(ImageList)
    width = 2
    height = (length + 1) / 2
    plt.figure(figsize=(40, 40))
    for i, image in enumerate(ImageList):
        plt.subplot(height, width, i + 1)
        plt.imshow(image, cmap)
        plt.autoscale(tight=True)
    plt.show()

#function to make filter mask on hls image
def colorFilter(hls):
    lower = np.array([0,210,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,50,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(hls, hls, mask=mask)
    return masked


def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55 * x), int(0.6 * y)], [int(0.45 * x), int(0.6 * y)]])
    #make numpy array size of img, but with zeros
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        # 3 channels for color depending on input image
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        # 1 channel for color depending on input image
        ignore_mask_color = 255

    #creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
    #returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#function to convert RGB to GRAY
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#function of Canny Edge Detector
def canny(img):
    return cv2.Canny(grayscale(img), 50, 120)


def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor = [0, 255, 0] # green
    leftColor = [255, 0, 0] # red

    # this is used to filter out the outlying lines that can affect the average
    # We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1 - y2) / (x1 - x2)
            if slope > 0.3:
                if x1 > 500:
                    yintercept = y2 - (slope * x2)
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else:
                    None
            elif slope < -0.3:
                if x1 < 600:
                    yintercept = y2 - (slope * x2)
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)
    # We use slicing operators and np.mean() to find the averages of the 30 previous frames
    # This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    # Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.65 * img.shape[0] - leftavgIntercept) / leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept) / leftavgSlope)
        right_line_x1 = int((0.65 * img.shape[0] - rightavgIntercept) / rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept) / rightavgSlope)
        pts = np.array([[left_line_x1, int(0.65 * img.shape[0])], [left_line_x2, int(img.shape[0])],
                        [right_line_x2, int(img.shape[0])], [right_line_x1, int(0.65 * img.shape[0])]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (0, 0, 255))
        cv2.line(img, (left_line_x1, int(0.65 * img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv2.line(img, (right_line_x1, int(0.65 * img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
        # I keep getting errors for some reason, so I put this here. Idk if the error still persists.
        pass


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # img is an output of canny() function (after Canny Edge Detection)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def linedetect(img):
    return hough_lines(img, 1, np.pi / 180, 10, 20, 100)


def weightSum(mask_img, original_img):
    return cv2.addWeighted(mask_img, 1, original_img, 0.8, 0)


img = cv2.imread('./Road/road2.jpeg')
ImageList.append(img)

#convert from BGR to HSL (hue, saturation, lightness)
hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
ImageList.append(hls_img)

filtered_img = colorFilter(hls_img)
ImageList.append(filtered_img)

roi_img = roi(filtered_img)
ImageList.append(roi_img)

canny_img = canny(roi_img)
ImageList.append(canny_img)

hough_img = linedetect(canny_img)
ImageList.append(hough_img)

result_img = weightSum(hough_img, img)
#ImageList.append(result_img)
plt.imshow(result_img, None)

showPictures(ImageList)