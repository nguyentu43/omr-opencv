import cv2
import utils
import numpy as np
from datetime import datetime
import imutils
from imutils import perspective

class OMR:

    rect_cols = 3
    top_offset = (10, 50)
    bottom_offset = (50, 10)
    min_circle_value = 700
    nums_row_in_col = 15

    debug=True
    resize_scale = 0.3
    draw_circle=True
    min_lightness = 250

    def __init__(self, img, debug=True):

        self.input_img = img
        self.debug = debug

        self.crop_img = []
        self.top_cnts = []
        self.bottom_cnts = []

        if self.input_img is None:
            exit('Input file not found')

    def crop(self):

        blurred = cv2.GaussianBlur(self.input_img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 70, 100)

        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) == 0:
            print('Can\'t find paper')
            return

        maxCnt = utils.max_contour(cnts)
        peri = cv2.arcLength(maxCnt[0], True)
        approx = cv2.approxPolyDP(maxCnt[0], 0.02 * peri, True)

        if len(approx) == 4:
            self.crop_img = perspective.four_point_transform(self.input_img, approx.reshape(4, 2))

        if len(self.crop_img) == 0:
            print('Can\'t find paper')

    def get_rects(self):

        if len(self.crop_img) == 0: return

        hsv = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        self.bw = 255 * np.uint8(v < self.min_lightness)

        if self.debug:
            cv2.imshow('bw', utils.resize(self.bw, self.resize_scale))

        rect_cnts = utils.find_contours(self.bw, 100, 300)
        if len(rect_cnts) < 6:
            print("Can't find boxs. Try again")
            return
            
        rect_cnts = utils.group_contours(rect_cnts, self.rect_cols)
        self.top_cnts = rect_cnts[0]
        self.bottom_cnts = rect_cnts[1]

    def find_information(self, info):

        if len(self.top_cnts) == 0: return

        information = ['class_name', 'student_number', 'test_code']
        cols_pattern = [4, 3, 3]
        try:
            info_index = information.index(info)
        except ValueError:
            print('Input info not found')
            return

        (x, y, w, h) = cv2.boundingRect(self.top_cnts[info_index])
        color = self.crop_img[y+self.top_offset[1]:y+h-self.top_offset[0], 
                    x+self.top_offset[0]:x+w-self.top_offset[0]]
        threshold = self.bw[y+self.top_offset[1]:y+h-self.top_offset[0], 
                    x+self.top_offset[0]:x+w-self.top_offset[0]]

        cnts = utils.find_contours(threshold, 30, 30, True)

        if self.debug:
            cv2.drawContours(color, cnts, -1, (255, 0, 255), 2)

        cols = cols_pattern[info_index]
        rows = len(cnts) // cols
        cnt_groups = utils.group_contours(cnts, cols)

        return_info = []
        for col in range(0, cols):
            mark_circle_cnts = []
            anwser = ''
            for row in range(0, rows):
                c = cnt_groups[row][col]
                (x, y, w, h) = cv2.boundingRect(c)
                total = cv2.countNonZero(threshold[y:y+h, x:x+w])
                if total > self.min_circle_value:
                    mark_circle_cnts.append(c)
                    if len(mark_circle_cnts) == 1: anwser = str(row)

            if len(mark_circle_cnts) > 1:
                print(info + ' invalid')
                if self.draw_circle: cv2.drawContours(color, mark_circle_cnts, -1, (0, 0, 255), 2)
            
            if self.draw_circle and len(mark_circle_cnts) == 1:
                cv2.drawContours(color, mark_circle_cnts, -1, (255, 0, 0), 2)
            if anwser != '': return_info.append(anwser)

        if len(return_info) == 0:
            return 'not found'
        return ''.join(return_info)

    def mark_test(self, anwser_keys):

        if len(self.bottom_cnts) == 0: return

        anwsers = []
        question_index = 0
        total_question = len(anwser_keys)
        mark = 0

        for i in range(0, len(self.bottom_cnts)):
            (x, y, w, h) = cv2.boundingRect(self.bottom_cnts[i])
            color = self.crop_img[y+self.bottom_offset[1]:y+h-self.bottom_offset[1], 
                        x+self.bottom_offset[0]:x+w-self.bottom_offset[1]]
            threshold = self.bw[y+self.bottom_offset[1]:y+h-self.bottom_offset[1], 
                        x+self.bottom_offset[0]:x+w-self.bottom_offset[1]]
            cnts = utils.find_contours(threshold, 30, 30, True)

            if self.debug:
                cv2.drawContours(color, cnts, -1, (255, 0, 255), 2)

            cols = 4
            rows = len(cnts) // cols
            cnt_groups = utils.group_contours(cnts, cols)
            
            for row in range(0, rows):
                mark_circle_cnts = []
                anwser = ''
                question_index += 1

                if question_index > total_question:
                    break

                for col in range(0, cols):
                    c = cnt_groups[row][col]
                    (x, y, w, h) = cv2.boundingRect(c)
                    total = cv2.countNonZero(threshold[y:y+h, x:x+w])

                    if total > self.min_circle_value:
                        mark_circle_cnts.append(c)
                        anwser = chr(col + 65)

                if len(mark_circle_cnts) > 1:
                    print('Question {} invalid'.format(question_index))
                    anwsers.append('invalid')
                    if self.draw_circle: cv2.drawContours(color, mark_circle_cnts, -1, (0, 0, 255), 2)
                elif len(mark_circle_cnts) == 1:
                    print('Question {} anwser {}'.format(question_index, anwser))
                    anwsers.append(anwser)
                    if anwser_keys[row] == anwser: 
                        mark += 1
                    if self.draw_circle: cv2.drawContours(color, mark_circle_cnts, -1, (255, 0, 0), 2)
                else:
                    print('Question {} empty'.format(question_index))
                    anwsers.append('empty')

                if self.draw_circle:
                    index_anwser = ord(anwser_keys[row]) - 65
                    cv2.drawContours(color, [cnt_groups[row][index_anwser]], -1, (0, 255, 0), 2)

            if question_index > total_question:
                break

        if len(anwsers) == 0 or len(anwsers) != total_question:
            print("Can't mark test")
        return round(mark * 10 / total_question, 1)

    def run(self, anwser_keys):

        self.crop()
        self.get_rects()

        result = []

        if len(self.crop_img) > 0 and len(self.top_cnts) > 0 and len(self.bottom_cnts) > 0:

            class_name = self.find_information('class_name')
            student_number = self.find_information('student_number')
            test_code = self.find_information('test_code')

            mark = self.mark_test(anwser_keys)

            result = self.crop_img.copy()

            x = int(result.shape[1] * 0.65)
            y = 60

            cv2.putText(result, 'Class Name: ' + class_name, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(result, 'Student Number: ' + student_number, (x, y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(result, 'Test Code: ' + test_code, (x, y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(result, 'Mark: ' + str( mark ), (x, y + 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(result, 'Date: ' + datetime.now().strftime('%c'), (x, y + 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return utils.resize(result, self.resize_scale)
        return result

img = cv2.imread('template-test.png')
omr = OMR(img, False)
omr.resize_scale = 0.5
result = omr.run(['A', 'A', 'B', 'C', 'D', 'C', 'A', 'A', 'C', 'D'])

while True:

    if len(result) > 0:
        cv2.imshow('Result', result)
        k = cv2.waitKey(1)
        if k == ord('s'):
            filename = 'results/result-{}.jpg'.format(datetime.now().strftime('%c').replace(':', '.'))
            cv2.imwrite(filename, result)
            break
    if k == 27:
        break

cv2.destroyAllWindows()

# cam = cv2.VideoCapture(0)

# while True:
#     ret, frame = cam.read()
#     omr = OMR(frame, False)
#     omr.resize_scale = 0.5
#     result = omr.run(['A', 'A', 'B', 'C', 'D', 'C', 'A', 'A', 'C', 'D'])
#     if len(result) > 0: cv2.imshow('check', result)
#     cv2.imshow('cam', frame)
#     key = cv2.waitKey(100)
#     if key == 27:
#         break
#     if key == ord('s') and len(result) > 0:
#         filename = 'results/result-{}.jpg'.format(datetime.now().strftime('%c').replace(':', '.'))
#         cv2.imwrite(filename, result)
#         break
# cam.release()
# cv2.destroyAllWindows()
