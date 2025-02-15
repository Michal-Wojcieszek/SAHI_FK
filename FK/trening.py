from ultralytics import YOLO

if __name__ == '__main__':
    modeln = YOLO(r"models/yolo11n.pt") 
    resultsn = modeln.train(data=r"SODA-D/soda_d_slice.yaml", epochs=100, imgsz=640, device=[0] , batch=64, resume=False, save_period = 1, plots=True)
    resultsn = modeln.val()
