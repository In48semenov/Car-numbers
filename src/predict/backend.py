from os.path import exists

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .detection import YoloInference, FasterRCNNInference, MTCNNInference
from .ocr import EasyOCRModel, EasyOCRCustom, LPRNETInference


class Inference:
    def __init__(
        self,
        detect_model: str = "yolo",
        ocr_model: str = "easyocr",
        plot: bool = True,
        font_path: str = "./src/utils/fonts/NotoSans.ttc"
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
            raise ValueError('Given detect model not found')

        if ocr_model == "easyocr":
            self.ocr_model = EasyOCRModel()
        elif ocr_model == "easyocr_custom":
            self.ocr_model = EasyOCRCustom()
        elif ocr_model == "lprnet":
            self.ocr_model = LPRNETInference()
        else:
            raise ValueError('Given detect model not found')

        self.font_path = font_path
        self.plot = plot

    def _get_number(self, image: Image, bbox):
        return np.asarray(image.crop((bbox[0], bbox[1], bbox[2], bbox[3])))

    def _demonstration(self, image: Image, bbox, text: str) -> Image:
        myFont = ImageFont.truetype(self.font_path, 24)
        draw = ImageDraw.Draw(image)
        draw.line((bbox[0], bbox[1], bbox[2], bbox[1]), fill=128, width=4)
        draw.line((bbox[2], bbox[1], bbox[2], bbox[3]), fill=128, width=4)
        draw.line((bbox[2], bbox[3], bbox[0], bbox[3]), fill=128, width=4)
        draw.line((bbox[0], bbox[3], bbox[0], bbox[1]), fill=128, width=4)
        draw.text((bbox[0], bbox[3]), text, fill=(255, 128, 0), font=myFont)
        return image

    def __call__(self, path_to_image: str):
        """
        path_to_image (str): path to image
        """
        image_orig = Image.open(path_to_image)
        result_number = []
        vis_image = None
        if self.plot:
            vis_image = image_orig.copy()

        detect_results = self.detect_model(image_orig)

        for bbox in detect_results:
            img_number = self._get_number(image_orig, bbox)

            text_recognition = self.ocr_model(img_number)
            result_number.append(text_recognition)
            if self.plot:
                if text_recognition is None:
                    text_recognition = "Не считано."
                vis_image = self._demonstration(vis_image, bbox, text_recognition)

        return result_number, vis_image
