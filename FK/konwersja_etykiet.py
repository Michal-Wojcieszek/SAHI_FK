from pathlib import Path
import globox


def main() -> None:
  txt                = r"SODA-D/foo/labels"
  zdjecia            = r"SODA-D/Slices640/images"
  anotacje           = r"SODA-D/Slices640/train_shifted_640.json"

  coco = globox.AnnotationSet.from_coco(
    file_path= anotacje,
    verbose=True
  )

  coco.show_stats()

  coco2 = coco.map_labels({"people": "0", "rider": "1","bicycle": "2", "motor": "3", "vehicle": "4","traffic-sign": "5", "traffic-light": "6", "traffic-camera": "7","warning-cone": "8", "ignore": "9"})

# {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9"}

  coco2.save_yolo_v5(
     txt,
     verbose=True
  )

if __name__ == "__main__":
  main()
