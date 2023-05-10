import cv2
import numpy as np


class LicensePlateProcessing:
    def __init__(self):
        self.max_w = 1280
        self.max_h = 720
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.canny_threshold1 = 40
        self.canny_threshold2 = 60
        self.blur_size = 5
        self.filter_di = 16
        self.filter_sig_color = 60
        self.filter_sig_space = 200
        self.min_plate_ratio = 2.0
        self.max_plate_ratio = 6.0
        self.all_contours_color = (0, 255, 0)
        self.plate_contours_color = (0, 0, 255)
        self.min_plate_width = 1/3
        self.min_plate_height = 1/8

    def fit_to_screen(self, img):
        if img.shape[0] > self.max_h or img.shape[1] > self.max_w:
            ratio = min(self.max_w / img.shape[1], self.max_h / img.shape[0])
            new_w = int(img.shape[1] * ratio)
            new_h = int(img.shape[0] * ratio)
            img = cv2.resize(img, (new_w, new_h))
        return img

    def image_filtering(self, img):
        blurred_img = cv2.GaussianBlur(img, (self.blur_size, self.blur_size), 0)
        filtered_img = cv2.bilateralFilter(blurred_img, self.filter_di, self.filter_sig_color, self.filter_sig_space)
        return filtered_img

    def find_contours(self, img):
        edges = cv2.Canny(img, self.canny_threshold1, self.canny_threshold2)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        dilated = cv2.dilate(closing, self.kernel, iterations=1)
        eroded = cv2.erode(dilated, self.kernel, iterations=1)
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_license_plate(self, contours, scaled_img):
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / float(h)
            if self.is_valid_plate_ratio(ratio, w, h, scaled_img):
                valid_contours.append(contour)
        if valid_contours:
            max_contour = max(valid_contours, key=cv2.contourArea)
            return [max_contour]
        return None

    def is_valid_plate_ratio(self, ratio, w, h, scaled_img):
        min_w = scaled_img.shape[1] * self.min_plate_width
        min_h = scaled_img.shape[0] * self.min_plate_height
        return self.min_plate_ratio <= ratio <= self.max_plate_ratio and w > min_w and h > min_h

    def perform_processing(self, img: np.ndarray) -> str:
        print(f'image.shape: {img.shape}')
        scaled_img = self.fit_to_screen(img)
        gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
        filtered_img = self.image_filtering(gray_img)
        contours = self.find_contours(filtered_img)
        plate_contour = self.find_license_plate(contours, scaled_img)
        img_with_contours = cv2.drawContours(scaled_img.copy(), contours, -1, self.all_contours_color, 2)
        img_with_plate_contours = cv2.drawContours(scaled_img.copy(), plate_contour, -1, self.plate_contours_color, 3)
        cv2.imshow("Processed Image", img_with_plate_contours)
        cv2.waitKey(0)
        return 'PO12345'
