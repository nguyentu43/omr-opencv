import cv2
import imutils
from imutils import contours

def resize(img, scale = 0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def group_contours(cnts, cols):
    if len(cnts) == 0: return []
    rows = len(cnts) // cols
    sort_cnts = contours.sort_contours(cnts, 'top-to-bottom')[0]
    return [contours.sort_contours(sort_cnts[i * cols:(i + 1) * cols])[0] for i in range(0, rows)]

def find_contours(img, filterW = 100, filterH = 100, circle=False):
    match_cnts = []
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (_, __, w, h) = cv2.boundingRect(c)
        if w >= filterW and h >= filterH:
            if circle == False:
                match_cnts.append(c)
            elif abs(w - h) <= 4: 
                match_cnts.append(c)
            else: continue
    return match_cnts

def imshow_contour(img, contour, window_name):
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.imshow(window_name, img[y:y+h, x:x+w])

def max_contour(cnts):
    maxCnt = (cnts[0], cv2.boundingRect(cnts[0]))
    for c in cnts[1:]:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > maxCnt[1][2] and h > maxCnt[1][3]:
            maxCnt = (c, (x, y, w, h))
    return maxCnt