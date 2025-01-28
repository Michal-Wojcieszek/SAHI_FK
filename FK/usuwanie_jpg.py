import os

def usun_jpg(jpg_folder, txt_folder):
    try:
        txt_files = {os.path.splitext(filename)[0] for filename in os.listdir(txt_folder) if filename.endswith('.txt')}
        
        for filename in os.listdir(jpg_folder):
            if filename.endswith('.jpg'):
                jpg_name = os.path.splitext(filename)[0]  
                jpg_path = os.path.join(jpg_folder, filename)
                
                if jpg_name not in txt_files:
                    os.remove(jpg_path)
                    print(f"Usunięto plik: {jpg_path}")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

jpg_folder_path = "SODA-D/valid_slice/images/"
txt_folder_path = "SODA-D/valid_slice/labels"

usun_jpg(jpg_folder_path, txt_folder_path)
