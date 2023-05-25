import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import string


class CharsGenerator:
    def __init__(self):
        # font setting
        self.char_height = 200
        self.font_path: str = f"{os.path.dirname(__file__)}/font/DIN_1451_Mittelschrift_Regular.ttf"
        self.font = ImageFont.truetype(self.font_path, self.char_height)
        self.background_color = (255, 255, 255)
        self.text_color = (0, 0, 0)

        self.license_plate_chars = list(string.ascii_uppercase + string.digits)

    def generate_chars_imgs(self) -> list[list[str, np.ndarray]]:
        chars_and_images = []
        for char in self.license_plate_chars:
            img = self.create_img_with_char(char, self.font)
            only_char = self.extract_char(img)
            chars_and_images.append([char, only_char])
        return chars_and_images

    def create_img_with_char(self, char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
        img = Image.new("RGB", (self.char_height, self.char_height), self.background_color)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=font, fill=self.text_color)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        return img

    def extract_char(self, img: np.ndarray):
        canny = cv2.Canny(img, 0, 255)
        canny = self.filter_edges(canny)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cutted_img = self.crop_imges(img, max(contours, key=cv2.contourArea))
        return cutted_img

    def filter_edges(self, edges):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    def crop_img(self, img, contours):
        chars = []
        for contour in contours:
            chars.append(self.crop_imges(img, contour))
        return chars

    def crop_imges(self, img, contour):
        (x, y, w, h) = cv2.boundingRect(contour)
        return img[y: y + h, x: x + w]
