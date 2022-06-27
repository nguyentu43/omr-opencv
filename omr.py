import cv2
import utils
import numpy as np
from datetime import datetime

class OMR:

    rect_cols = 3
    top_offset = (5, 50)
    bottom_offset = (50, 5)
    min_circle_value = 700
    nums_row_in_col = 15

    debug=True
    resize_scale = 0.3
    draw_circle=True

    def __init__(self, img_file, debug=True):

        self.input_img = cv2.imread(img_file)
        self.debug = debug
        if self.input_img is None:
            exit('Input file not found')

    def crop(self):
        pass

    def find_information(self, info):

        information = ['class_name', 'student_number', 'test_code']
        cols_pattern = [4, 3, 3]
        try:
            info_index = information.index(info)
        except ValueError:
            print('Input info not found')
            return

        (x, y, w, h) = cv2.boundingRect(self.top_cnts[info_index])
        color = self.input_img[y+self.top_offset[1]:y+h-self.top_offset[0], 
                    x+self.top_offset[0]:x+w-self.top_offset[0]]
        threshold = self.bw[y+self.top_offset[1]:y+h-self.top_offset[0], 
                    x+self.top_offset[0]:x+w-self.top_offset[0]]
        cnts = utils.find_contours(threshold, 30, 30)

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
            
            if len(mark_circle_cnts) == 0:
                return_info.append('')
            else:
                return_info.append(anwser)

        if len(return_info) == 0:
            print("Can't find " + info)
        return ''.join(return_info)

    def mark_test(self, anwser_keys):

        anwsers = []
        question_index = 0
        total_question = len(anwser_keys)
        point = 0

        for i in range(0, len(self.bottom_cnts)):
            (x, y, w, h) = cv2.boundingRect(self.bottom_cnts[i])
            color = self.input_img[y+self.bottom_offset[1]:y+h-self.bottom_offset[1], 
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
                        point += 1
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
        return round(point * 10 / total_question, 1)

    def run(self, anwser_keys):

        self.crop()

        hsv = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        self.bw = 255 * np.uint8(v < 150)

        if self.debug:
            cv2.imshow('BW', utils.resize(self.bw, self.resize_scale))
            cv2.waitKey(0)

        rect_cnts = utils.find_contours(self.bw, 100, 300)
        if len(rect_cnts) < 6:
            print("Can't find boxs. Try again")
            return
            
        rect_cnts = utils.group_contours(rect_cnts, self.rect_cols)
        self.top_cnts = rect_cnts[0]
        self.bottom_cnts = rect_cnts[1]

        class_name = self.find_information('class_name')
        student_number = self.find_information('student_number')
        test_code = self.find_information('test_code')

        point = self.mark_test(anwser_keys)

        x = int(self.input_img.shape[1] * 0.65)
        y = 60

        cv2.putText(self.input_img, 'Class Name: ' + class_name, (x, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(self.input_img, 'Student Number: ' + student_number, (x, y + 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(self.input_img, 'Test Code: ' + test_code, (x, y + 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(self.input_img, 'Point: ' + str( point ), (x, y + 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(self.input_img, 'Date: ' + datetime.now().strftime('%c'), (x, y + 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        resize = utils.resize(self.input_img, self.resize_scale)
        while True:
            cv2.imshow('Result', resize)
            k = cv2.waitKey(1)
            if k == ord('s'):
                filename = 'results/result-{}.jpg'.format(datetime.now().strftime('%c').replace(':', '.'))
                cv2.imwrite(filename, self.input_img)
                break
            if k == 27:
                break
        cv2.destroyAllWindows()

omr = OMR('template-test.png')
omr.run(['A', 'A', 'B', 'C', 'D', 'C', 'A', 'A', 'C', 'D'])

