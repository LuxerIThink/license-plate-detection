import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import string


class CharsImgGenerator:
    def __init__(self, char_height: int = None, font_path: str = None):
        # font setting
        self.char_height = char_height or 200
        self.font_path: str = font_path or f"{os.path.dirname(__file__)}/font/DIN_1451_Mittelschrift_Regular.ttf"
        self.font = ImageFont.truetype(self.font_path, self.char_height)
        self.bg_color = (255, 255, 255)
        self.text_color = (0, 0, 0)
        self.license_plate_chars = list(string.ascii_uppercase + string.digits)

    def generate_chars_imgs(self) -> list[list[str, np.ndarray]]:
        chars_and_images = []
        for char in self.license_plate_chars:
            img = self.create_char_img(char, self.font)
            chars_and_images.append([char, img])
        return chars_and_images

    def create_char_img(self, char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
        img = Image.new("RGB", (self.char_height, self.char_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=font, fill=self.text_color)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img = self.remove_margins(img)
        return img

    def remove_margins(self, img: np.ndarray) -> np.ndarray:
        canny = cv2.Canny(img, 0, 255)
        canny = self.filter_edges(canny)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cropped_img = self.crop_img(img, max(contours, key=cv2.contourArea))
        return cropped_img

    @staticmethod
    def filter_edges(edges: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    def crop_images(self, img: np.ndarray, contours: np.ndarray) -> list[np.ndarray]:
        chars = []
        for contour in contours:
            chars.append(self.crop_img(img, contour))
        return chars

    @staticmethod
    def crop_img(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
        (x, y, w, h) = cv2.boundingRect(contour)
        return img[y: y + h, x: x + w]
