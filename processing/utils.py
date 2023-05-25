import cv2
import numpy as np

from .chars_generator import CharsGenerator


class LicensePlateProcessing:
    def __init__(self):
        # image max size
        self.target_width: int = 960

        # image processing
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.gaussian_blur: tuple[int, int] = (7, 5)

        # img filtering
        self.filter_di: int = 15
        self.filter_sig_color: int = 25
        self.filter_sig_space: int = 50

        # plate requirements
        self.second_plate_ratio: float = 0.7
        self.min_plate_ratio: float = 2
        self.max_plate_ratio: float = 6
        self.approx_poly_dp: float = 0.015

        # plate settings
        self.plate_width: float = 1000
        self.plate_height: float = 200

        self.chars_generator = CharsGenerator()
        self.chars_generator.generate_chars_imgs()

    def perform_processing(self, img: np.ndarray) -> str:
        scaled_img = self.resize_img(img)
        gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
        filtered_img = self.img_filtering(gray_img)
        img_contours = self.find_contours(filtered_img)
        plate_contour = self.find_plate_approx(img_contours)
        if plate_contour is None:
            return ""
        plate_img = self.cut_plate_img(scaled_img, plate_contour)
        # img_with_contours = cv2.drawContours(
        #     scaled_img.copy(), img_contours, -1, (0, 0, 255), 3
        # )
        # img_with_plate_contours = cv2.drawContours(
        #     img_with_contours, plate_contour, -1, (0, 255, 0), 3
        # )
        # img_plate = cv2.drawContours(plate_img, plate_contour, -1, (0, 255, 0), 3)
        # cv2.imshow("Processed Image", img_with_plate_contours)
        # cv2.waitKey(0)
        return ""

    def resize_img(self, img: np.ndarray) -> np.ndarray:
        if img.shape[1] != self.target_width:
            ratio = self.target_width / img.shape[1]
            new_width = self.target_width
            new_height = int(img.shape[0] * ratio)
            img = cv2.resize(img, (new_width, new_height))
        return img

    def img_filtering(self, img: np.ndarray) -> np.ndarray:
        img = cv2.bilateralFilter(
            img, self.filter_di, self.filter_sig_color, self.filter_sig_space
        )
        img = cv2.GaussianBlur(img, self.gaussian_blur, 0)
        return img

    def find_contours(self, img: np.ndarray) -> np.ndarray:
        edges = self.get_edges(img)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_edges(self, img: np.ndarray) -> np.ndarray:
        edges = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 3
        )
        edges = self.edit_edges(edges)
        return edges

    def edit_edges(self, edges: np.ndarray) -> np.ndarray:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        edges = cv2.dilate(edges, self.kernel, iterations=1)
        edges = cv2.erode(edges, self.kernel, iterations=1)
        return edges

    def find_plate_approx(self, approxes: np.ndarray) -> list:
        approxes = self.filter_contours(approxes)
        plate_approx = self.choose_plate(approxes)
        return plate_approx

    def approx_contour(self, contour: list[np.ndarray]) -> np.ndarray:
        return cv2.approxPolyDP(
            contour, self.approx_poly_dp * cv2.arcLength(contour, True), True
        )

    def filter_contours(self, contours: np.ndarray) -> list[np.ndarray]:
        approxes = []
        for contour in contours:
            approx = self.approx_contour(contour)
            if len(approx) == 4 and self.is_valid_ratio(approx):
                approxes.append(approx)
        return approxes

    def is_valid_ratio(self, approx: np.ndarray) -> bool:
        side_lengths = [cv2.norm(approx[i % 4] - approx[(i + 1) % 4]) for i in range(4)]
        ratio = max(side_lengths) / min(side_lengths)
        return self.min_plate_ratio < ratio < self.max_plate_ratio

    def choose_plate(
        self, contours: list[np.ndarray]
    ) -> list[np.ndarray] | None:
        contours.sort(key=cv2.contourArea, reverse=True)
        if contours:
            if len(contours) >= 2:
                if cv2.contourArea(
                    contours[1]
                ) > self.second_plate_ratio * cv2.contourArea(contours[0]):
                    return [contours[1]]
            return [contours[0]]
        return None

    def cut_plate_img(self, img: np.ndarray, approx: list) -> np.ndarray:
        new_polygon = np.array(self.sort_points_clockwise(approx[0]), dtype=np.float32)
        transformation_matrix = cv2.getPerspectiveTransform(
            new_polygon, self.get_target_rect()
        )
        transformed_image = cv2.warpPerspective(
            img, transformation_matrix, (self.plate_width, self.plate_height)
        )
        return transformed_image

    def sort_points_clockwise(self, points: list) -> list:
        sorted_by_y = sorted(points, key=lambda point: point[0][1])
        top_order = sorted(sorted_by_y[:2], key=lambda point: point[0][0])
        bottom_order = sorted(
            sorted_by_y[2:], key=lambda point: point[0][0], reverse=True
        )
        sorted_points = top_order + bottom_order
        return sorted_points

    def get_target_rect(self) -> np.ndarray:
        target_rect = np.array(
            [
                [0, 0],
                [self.plate_width, 0],
                [self.plate_width, self.plate_height],
                [0, self.plate_height],
            ],
            dtype=np.float32,
        )
        return target_rect
