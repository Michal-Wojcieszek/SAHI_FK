import argparse
import numpy as np
import os
from typing import List, Dict
import my_utils
import statistics
import logging
logger = logging.getLogger(__name__)
import tqdm

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sahi_setup_eval import sahi_fun_eval
import my_utils
from datetime import datetime

class COCOEvaluator:
    """
    Evaluates object detection models using COCO metrics.
    """

    def __init__(self, model_path: str, coco_json: str, image_dir: str, confidence_threshold: float = 0.3,
                 slice_height: int = 640, slice_width: int = 640, overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2, output_dir="stats.txt"):
        """
        Initialises the evaluator with model and dataset parameters.

        Args:
            model_path (str): Path to the trained model file.
            coco_json (str): Path to the COCO format annotations JSON file.
            image_dir (str): Directory containing the evaluation image set.
            confidence_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.3.
            slice_height (int, optional): Height of the slices for prediction. Defaults to 640.
            slice_width (int, optional): Width of the slices for prediction. Defaults to 640.
            overlap_height_ratio (float, optional): Overlap ratio between slices in height. Defaults to 0.2.
            overlap_width_ratio (float, optional): Overlap ratio between slices in width. Defaults to 0.2.
        """
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cuda:0",  # Uncomment this to force CPU usage
        )
        self.coco_json = coco_json
        self.image_dir = image_dir
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.model_path=model_path
        self.output_dir = output_dir

    def evaluate(self) -> Dict[str, float]:
        """
        Performs the evaluation of the model against the given dataset and computes COCO metrics.

        Returns:
            Dict[str, float]: A dictionary containing computed metrics.
        """
        coco = Coco.from_coco_dict_or_path(self.coco_json)
        coco.add_categories_from_coco_category_list(my_utils.coco_category_list)
        pycoco = COCO(self.coco_json)
        predictions = []
        category_to_id = {category.name: category.id for category in coco.categories}

        for image_info in tqdm.tqdm(coco.images, position=0, leave=True):
            image_path = os.path.join(self.image_dir, image_info.file_name)

            prediction_result = sahi_fun_eval(
                self.model,
                jpg_path=image_path, 
                NMS = "NMS",
                podzial=my_utils.podzial, #6
                nakladanie=my_utils.nakladanie, #0.4 
                czulosc_IOU=0.45
            )
            for pred in prediction_result.object_prediction_list:
                #if pred.category.name != "ignore":
                    predictions.append({
                        "image_id": image_info.id,
                        "category_id": category_to_id[pred.category.name],
                        "bbox": [
                            pred.bbox.minx, pred.bbox.miny,
                            pred.bbox.maxx - pred.bbox.minx, pred.bbox.maxy - pred.bbox.miny
                        ],
                        "score": pred.score.value,
                    })

        pycoco_pred = pycoco.loadRes(predictions) # type: ignore
        coco_eval = COCOeval(pycoco, pycoco_pred, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        with open(self.output_dir, "a") as f:
                f.write(f"{[i * 100 for i in coco_eval.stats]}\n")

        metrics = {
            "Average Precision": np.mean(coco_eval.eval['precision'][:, :, :, 0, -1], axis=(0, 1, 2)),
            "Average Recall": np.mean(coco_eval.eval['recall'][:, :, 0, -1], axis=(0, 1)),
            "mAP at IoU=50": np.mean(coco_eval.eval['precision'][0, :, :, 0, 2]),
            "mAP at IoU=50-95": np.mean(coco_eval.eval['precision'][0, :, :, 0, :])
        }

        print("Czas: ", statistics.fmean(my_utils.czasy_eval))
        print("Usunięte wycinki: ", my_utils.liczba_wycinkow - my_utils.pozostale_wyc)
        print("Wszystkie wycinki: ", my_utils.liczba_wycinkow)
        print(f"Procent usuniętych: {100*(my_utils.liczba_wycinkow-my_utils.pozostale_wyc)/my_utils.liczba_wycinkow}%")
        # print(my_utils.czasy_eval)
        
        with open(self.output_dir, "a") as f:
            f.write(f"Czas:\n{statistics.fmean(my_utils.czasy_eval)}\n")
            f.write(f"Usunięte wycinki:\n{my_utils.liczba_wycinkow - my_utils.pozostale_wyc}\n")
            f.write(f"Wszystkie wycinki:\n{my_utils.liczba_wycinkow}\n")
            f.write(f"Procent usuniętych:\n{100*(my_utils.liczba_wycinkow-my_utils.pozostale_wyc)/my_utils.liczba_wycinkow}\n")

        return metrics

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluates a YOLOv8 model using COCO metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to the COCO format annotations JSON file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the evaluation image set.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    evaluator = COCOEvaluator(
        model_path=args.model_path,
        coco_json=args.coco_json,
        image_dir=args.image_dir
    )
    metrics = evaluator.evaluate()
    print("Evaluation metrics:", metrics)

def ewaluacja(model_path: str, coco_json: str, image_dir: str, conf_th = 0.2, output_dir: str = "stats.txt") -> None:

    with open(output_dir, "a") as f:
        f.write(f"{datetime.now()}\n")
        f.write(f"{model_path} {coco_json} {image_dir} conf_th: {conf_th}\n")
        f.write(f"podzial:\n{my_utils.podzial}\nnakladanie:\n{my_utils.nakladanie}\nfiltrowanie:\n{my_utils.filtruj_puste_wycinki}, canny th1: {my_utils.canny_th1}, canny th2: {my_utils.canny_th2}\n")

    evaluator = COCOEvaluator(
        model_path=model_path,
        coco_json=coco_json,
        image_dir=image_dir,
        confidence_threshold=conf_th,
        output_dir=output_dir
    )
    evaluator.evaluate()
    with open(output_dir, "a") as f:
        f.write(f"{datetime.now()}\n\n")

    my_utils.pozostale_wyc = 0
    my_utils.liczba_wycinkow = 0
    print(len(my_utils.czasy_eval))
    # print(my_utils.czasy_eval)
    my_utils.czasy_eval.clear()