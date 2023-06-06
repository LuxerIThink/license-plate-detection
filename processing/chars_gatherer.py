import cv2
import numpy as np

from .chars_img_generator import CharsImgGenerator

class CharsGatherer:
    def __init__(self, plate_height):

        # img filtering
        self.filter_di: int = 15
        self.filter_sig_color: int = 25
        self.filter_sig_space: int = 50
        self.gaussian_blur: tuple[int, int] = (5, 5)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.threshold1 = 100
        self.threshold2 = 200

        # find chars dp
        self.approx_poly_dp = 0.008

        # plate dims
        self.plate_height = plate_height

        # contour filter
        self.min_ratio = 0.3
        self.max_ratio = 0.9
        self.char_to_plate = 0.6

        # features approx dp
        self.fix_mistakes_table: dict = {
            "Z": ("7", 1.4),
            "C": ("G", 1.1),
            "S": ("5", 1.5),
            "J": ("4", 1.2),
            "Y": ("1", 1.3),
            "L": ("K", 1.2),
            "K": ("W", 1.2),
            "0": ("G", 1.1),
            "R": ("W", 1.1),
            "O": ("0", 1.0083),
            "N": ("W", 1.1),
        }

        self.chars_generator = CharsImgGenerator()
        self.template_chars = self.chars_generator.generate_chars_imgs()

    def get_str(self, plate_img: np.ndarray):
        plate_img = self.filter_img(plate_img)
        contours = self.find_edges(plate_img)
        chars_contours = self.find_chars_contours(contours)
        chars_images = self.crop_images(plate_img, chars_contours)
        string = self.get_string(chars_images)
        return string

    def filter_img(self, img: np.ndarray) -> np.ndarray:
        img = cv2.bilateralFilter(img, self.filter_di, self.filter_sig_color, self.filter_sig_space)
        img = cv2.GaussianBlur(img, self.gaussian_blur, 0)
        return img

    def find_edges(self, img: np.ndarray) -> list:
        edges = cv2.Canny(img, self.threshold1, self.threshold2)
        edges = self.edit_edges(edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    def edit_edges(self, edges: np.ndarray) -> np.ndarray:
        edges = cv2.dilate(edges, self.kernel, iterations=1)
        edges = cv2.erode(edges, self.kernel, iterations=1)
        return edges

    def find_chars_contours(self, contours: list) -> list | None:
        approxes = []
        for contour in contours:
            approx = self.is_valid_contour(contour)
            if approx is not None:
                approxes.append(approx)
        if approxes is None:
            return None
        approxes = sorted(approxes, key=lambda c: cv2.boundingRect(c)[0])
        approxes = self.remove_approx_duplicates(approxes)
        return approxes

    def is_valid_contour(self, contour: np.ndarray) -> np.ndarray | None:
        approx, char_to_plate, ratio = self.get_contours_features(contour)
        if self.min_ratio < ratio < self.max_ratio and char_to_plate > self.char_to_plate and len(approx) > 5:
            return approx
        return None

    def get_contours_features(self, contour: np.ndarray) -> tuple[np.ndarray, float, float]:
        x, y, width, height = cv2.boundingRect(contour)
        ratio = float(width) / height
        char_to_plate = float(height) / self.plate_height
        approx = cv2.approxPolyDP(contour, self.approx_poly_dp * cv2.arcLength(contour, True), True)
        return approx, char_to_plate, ratio

    def remove_approx_duplicates(self, approxs: list) -> list:
        new_approxes = []
        previous_approx = None
        for approx in approxs:
            if previous_approx is None or not self.is_approx_inside(approx, previous_approx):
                new_approxes.append(approx)
            previous_approx = approx
        return new_approxes

    def is_approx_inside(self, current_approx: list, previous_approx: list) -> bool:
        current_rect = cv2.boundingRect(current_approx)
        previous_rect = cv2.boundingRect(previous_approx)
        if self.is_rect_inside(current_rect, previous_rect) or self.is_rect_inside(previous_rect, current_rect):
            return True
        return False

    @staticmethod
    def add_margin_to_rect(rect: list, margin_percentage: float) -> list:
        margin = rect[2] * margin_percentage  # Calculate margin based on width
        rect_with_margin = [
            rect[0] - margin,  # Left side with margin
            rect[1] - margin,  # Top side with margin
            rect[2] + 2 * margin,  # Increased width by margin on both sides
            rect[3] + 2 * margin,  # Increased height by margin on both sides
        ]
        return rect_with_margin

    def is_rect_inside(self, inner_rect: list, outer_rect: list) -> bool:
        inner_rect = self.add_margin_to_rect(inner_rect, 0.4)
        if (
                inner_rect[0] <= outer_rect[0]
                and inner_rect[1] <= outer_rect[1]
                and inner_rect[0] + inner_rect[2] >= outer_rect[0] + outer_rect[2]
                and inner_rect[1] + inner_rect[3] >= outer_rect[1] + outer_rect[3]
        ):
            return True
        return False

    def crop_images(self, img: np.ndarray, approx: list) -> list[np.ndarray]:
        chars_imgs = []
        for approx in approx:
            chars_imgs.append(self.crop_image(img, approx))
        return chars_imgs

    @staticmethod
    def crop_image(img: np.ndarray, contour: list) -> np.ndarray:
        cropped_image = img.copy()
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = cropped_image[y: y + h, x: x + w]
        _, treshold_image = cv2.threshold(cropped_image, 128, 255, cv2.THRESH_BINARY)
        return cropped_image

    def get_string(self, chars_imgs: list[np.ndarray]) -> str:
        string = ''
        for img in chars_imgs:
            char = self.get_char(img)
            string += char
        return string

    def get_char(self, img: np.ndarray) -> str:
        char = '-'
        best_match_prob = 1
        match_percentages = {}

        for char_str, char_img in self.template_chars:
            template_height, template_width, _ = char_img.shape
            img_resized = self.resize_image(img, template_width, template_height)
            match_prob = self.compare_images(img_resized, char_img)
            match_percentages[char_str] = match_prob

            if match_prob < best_match_prob:
                char = char_str
                best_match_prob = match_prob

        if char in self.fix_mistakes_table:
            skip_match, threshold = self.fix_mistakes_table[char]
            if (match_percentages[skip_match] / best_match_prob) < threshold:
                char = skip_match

        return char

    def resize_image(self, img: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(img, (width, height))

    def compare_images(self, img1: np.ndarray, img2: np.ndarray) -> str:
        result = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF_NORMED)
        _, match_prob, _, _ = cv2.minMaxLoc(result)
        return match_prob