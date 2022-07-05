import cv2
import numpy as np

from check_label import obj_diou,obj_iou

if __name__ == '__main__':
    image_size = 1200
    center_set = ((400, 400), (800, 800))
    detect_size = 400
    step = 5
    color_full = (0, 255, 0)

    white_back_00 = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    white_back_25 = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    white_back_50 = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    white_back_75 = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    white_back_90 = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    for x in range(0, image_size // 2 - detect_size // 2, step):
        for y in range(0, image_size // 2 - detect_size // 2, step):
            detect_set = ((x, y), (x + detect_size, y + detect_size))
            diou = obj_diou(center_set, detect_set)

            color = [max(int(diou * x), 0) for x in color_full]
            if diou > 0.90:
                cv2.rectangle(white_back_90, detect_set[0], detect_set[1], color, -1)
            elif diou > 0.75:
                cv2.rectangle(white_back_75, detect_set[0], detect_set[1], color, -1)
            elif diou > 0.5:
                cv2.rectangle(white_back_50, detect_set[0], detect_set[1], color, -1)
            elif diou > 0.25:
                cv2.rectangle(white_back_25, detect_set[0], detect_set[1], color, -1)
            else:
                cv2.rectangle(white_back_00, detect_set[0], detect_set[1], color, -1)

    cv2.rectangle(white_back_00, center_set[0], center_set[1], (128, 128, 128), 3)
    cv2.rectangle(white_back_25, center_set[0], center_set[1], (128, 128, 128), 3)
    cv2.rectangle(white_back_50, center_set[0], center_set[1], (128, 128, 128), 3)
    cv2.rectangle(white_back_75, center_set[0], center_set[1], (128, 128, 128), 3)
    cv2.rectangle(white_back_90, center_set[0], center_set[1], (128, 128, 128), 3)

    cv2.imwrite("white_back_00-25.png", white_back_00)
    cv2.imwrite("white_back_25-50.png", white_back_25)
    cv2.imwrite("white_back_50-75.png", white_back_50)
    cv2.imwrite("white_back_75-90.png", white_back_75)
    cv2.imwrite("white_back_90-100.png", white_back_90)
    