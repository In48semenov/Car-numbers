from os.path import exists

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detection import YoloInference, FasterRCNNInference, MTCNNInference
from .ocr import EasyOCRModel, EasyOCRCustom, LPRNETInference
from .transform import STNetInference


class Inference:
    def __init__(
        self,
        detect_model: str = "yolo",
        ocr_model: str = "easyocr",
        transform_model: str = None,
        plot: bool = True,
        font_path: str = "./src/utils/fonts/NotoSans.ttc",
    ):
        """
        detect_model (str): yolo / frcnn / mtcnn
        ocr_model (str): easyocr / easyocr_custom / lpr_custom
        """
        if detect_model == "yolo":
            self.detect_model = YoloInference()
        elif detect_model == "frcnn":
            self.detect_model = FasterRCNNInference()
        elif detect_model == "mtcnn":
            self.detect_model = MTCNNInference()
        else:
            raise ValueError("Given detect model not found")

        if ocr_model == "easyocr":
            self.ocr_model = EasyOCRModel()
        elif ocr_model == "easyocr_custom":
            self.ocr_model = EasyOCRCustom()
        elif ocr_model == "lprnet":
            self.ocr_model = LPRNETInference()
        else:
            raise ValueError("Given detect model not found")

        if transform_model == "stnet":
            self.transform_model = STNetInference()
        elif transform_model is None:
            self.transform_model = None
        else:
            raise ValueError("Given transform model not found")

        self.font_path = font_path
        self.plot = plot

        if transform_model is None:
            self.name_model = f"{detect_model}+{ocr_model}"
        else:
            self.name_model = f"{detect_model}+{transform_model}+{ocr_model}"

    def _get_number(self, image: Image, bbox):
        return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    def _demonstration(self, image: Image, bbox, text: str) -> Image:
        myFont = ImageFont.truetype(self.font_path, 24)
        draw = ImageDraw.Draw(image)
        draw.line((bbox[0], bbox[1], bbox[2], bbox[1]), fill=128, width=4)
        draw.line((bbox[2], bbox[1], bbox[2], bbox[3]), fill=128, width=4)
        draw.line((bbox[2], bbox[3], bbox[0], bbox[3]), fill=128, width=4)
        draw.line((bbox[0], bbox[3], bbox[0], bbox[1]), fill=128, width=4)
        draw.text((bbox[0], bbox[3]), text, fill=(255, 128, 0), font=myFont)
        return image

    def detect_by_image(self, image):
        image_orig = image
        result_number = []
        vis_image = None
        if self.plot:
            vis_image = image_orig.copy()

        detect_results = self.detect_model(image_orig)

        if detect_results is None:
            return result_number, vis_image

        for bbox in detect_results:
            img_number = np.asarray(self._get_number(image_orig, bbox))

            if self.transform_model is not None:
                img_number = np.asarray(self.transform_model(img_number))

            text_recognition = self.ocr_model(img_number)
            result_number.append(text_recognition)
            if self.plot:
                if text_recognition is None:
                    text_recognition = "Не считано."
                vis_image = self._demonstration(vis_image, bbox, text_recognition)

        return result_number, vis_image

    def detect_by_image_path(self, path_to_image: str):
        image_orig = Image.open(path_to_image)
        return self.detect_by_image(image_orig)

    def detect_by_video_path(self, path_to_video, save_result_path):
        vid_capture = cv2.VideoCapture(path_to_video)

        width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid_capture.get(cv2.CAP_PROP_FPS))

        output = cv2.VideoWriter(
            save_result_path,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps,
            (width, height),
        )

        while vid_capture.isOpened():
            ret, frame = vid_capture.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)

                _, result = self.detect_by_image(pil_img)

                open_cv_image = np.array(result)
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                output.write(open_cv_image)
            else:
                break

        vid_capture.release()
        output.release()
