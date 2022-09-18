import logging
import pkg_resources
import requests
import sys
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
from keras.models import load_model

from src.utils import load_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("expression")

NumpyRects = Union[np.ndarray, Sequence[Tuple[int, int, int, int]]]

PADDING = 40
SERVER_URL = "http://localhost:8501/v1/models/emotion_model:predict"


class Xpress(object):

    def __init__(
        self,
        cascade_file: str = None,
        mtcnn=False,
        tfserving: bool = False,
        scale_factor: float = 1.1,
        min_face_size: int = 50,
        min_neighbors: int = 5,
        offsets: tuple = (10, 10),
    ):

        self.__scale_factor = scale_factor
        self.__min_neighbors = min_neighbors
        self.__min_face_size = min_face_size
        self.__offsets = offsets
        self.tfserving = tfserving

        if cascade_file is None:
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        if mtcnn:
            try:
                from mtcnn.mtcnn import MTCNN
            except ImportError:
                raise Exception(
                    "MTCNN not installed, install it with pip install mtcnn"
                )
            self.__face_detector = "mtcnn"
            self._mtcnn = MTCNN()
        else:
            self.__face_detector = cv2.CascadeClassifier(cascade_file)

        self._initialize_model()

    def _initialize_model(self):
        if self.tfserving:
            self.__emotion_target_size = (64, 64)  # hardcoded for now
        else:

            emotion_model = pkg_resources.resource_filename(
                "src.express", "data/emotion_model.hdf5"
            )
            log.debug("Emotion model: {}".format(emotion_model))
            self.__emotion_classifier = load_model(emotion_model, compile=False)
            self.__emotion_classifier.make_predict_function()
            self.__emotion_target_size = self.__emotion_classifier.input_shape[1:3]
        return

    def _classify_emotions(self, gray_faces: np.ndarray) -> np.ndarray:  # b x w x h

        if self.tfserving:
            gray_faces = np.expand_dims(gray_faces, -1)  # to 4-dimensions
            instances = gray_faces.tolist()
            response = requests.post(SERVER_URL, json={"instances": instances})
            response.raise_for_status()

            emotion_predictions = response.json()["predictions"]
            return emotion_predictions
        else:
            return self.__emotion_classifier(gray_faces)

    @staticmethod
    def pad(image):

        row, col = image.shape[:2]
        bottom = image[row - 2 : row, 0:col]
        mean = cv2.mean(bottom)[0]

        padded_image = cv2.copyMakeBorder(
            image,
            top=PADDING,
            bottom=PADDING,
            left=PADDING,
            right=PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean],
        )
        return padded_image

    @staticmethod
    def depad(image):
        row, col = image.shape[:2]
        return image[PADDING : row - PADDING, PADDING : col - PADDING]

    @staticmethod
    def tosquare(bbox):

        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff
        if w != h:
            log.debug(f"{w} is not {h}")

        return (x, y, w, h)

    def find_faces(self, img: np.ndarray, bgr=True) -> list:

        if isinstance(self.__face_detector, cv2.CascadeClassifier):
            if bgr:
                gray_image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # assume gray
                gray_image_array = img

            faces = self.__face_detector.detectMultiScale(
                gray_image_array,
                scaleFactor=self.__scale_factor,
                minNeighbors=self.__min_neighbors,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(self.__min_face_size, self.__min_face_size),
            )
        elif self.__face_detector == "mtcnn":
            results = self._mtcnn.detect_faces(img)
            faces = [x["box"] for x in results]

        return faces

    @staticmethod
    def __preprocess_input(x, v2=False):
        x = x.astype("float32")
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def __apply_offsets(self, face_coordinates):

        x, y, width, height = face_coordinates
        x_off, y_off = self.__offsets
        x1 = x - x_off
        x2 = x + width + x_off
        y1 = y - y_off
        y2 = y + height + y_off
        return x1, x2, y1, y2

    @staticmethod
    def _get_labels():
        return {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral",
        }

    def detect_emotions(
        self, img: np.ndarray, face_rectangles: NumpyRects = None
    ) -> list:

        img = load_image(img)

        emotion_labels = self._get_labels()

        if face_rectangles is None:
            face_rectangles = self.find_faces(img, bgr=True)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = self.pad(gray_img)

        emotions = []
        gray_faces = []

        for face_coordinates in face_rectangles:
            face_coordinates = self.tosquare(face_coordinates)


            x1, x2, y1, y2 = self.__apply_offsets(face_coordinates)

            x1 += PADDING
            y1 += PADDING
            x2 += PADDING
            y2 += PADDING
            x1 = np.clip(x1, a_min=0, a_max=None)
            y1 = np.clip(y1, a_min=0, a_max=None)

            gray_face = gray_img[max(0, y1) : y2, max(0, x1) : x2]

            try:
                gray_face = cv2.resize(gray_face, self.__emotion_target_size)
            except Exception as e:
                log.warn("{} resize failed: {}".format(gray_face.shape, e))
                continue

            gray_face = self.__preprocess_input(gray_face, True)
            gray_faces.append(gray_face)

        if not len(gray_faces):
            return emotions  # no valid faces

        emotion_predictions = self._classify_emotions(np.array(gray_faces))

        for face_idx, face in enumerate(emotion_predictions):
            labelled_emotions = {
                emotion_labels[idx]: round(float(score), 2)
                for idx, score in enumerate(face)
            }

            emotions.append(
                dict(box=face_rectangles[face_idx], emotions=labelled_emotions)
            )

        self.emotions = emotions

        return emotions

    def top_emotion(
        self, img: np.ndarray
    ) -> Tuple[Union[str, None], Union[float, None]]:

        emotions = self.detect_emotions(img=img)
        top_emotions = [
            max(e["emotions"], key=lambda key: e["emotions"][key]) for e in emotions
        ]

        if len(top_emotions):
            top_emotion = top_emotions[0]
        else:
            return (None, None)
        score = emotions[0]["emotions"][top_emotion]

        return top_emotion, score


def parse_arguments(args):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Image filepath")
    return parser.parse_args()


def top_emotion():
    args = parse_arguments(sys.argv)
    expression = Xpress()
    top_emotion, score = expression.top_emotion(args.image)
    print(top_emotion, score)


def main():
    top_emotion()


if __name__ == "__main__":
    main()
