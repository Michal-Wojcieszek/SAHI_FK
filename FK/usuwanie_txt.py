import os

def usun_txt(txt_folder, jpg_folder):
    
    try:
        for txt_filename in os.listdir(txt_folder):
            if txt_filename.endswith('.txt'):  
                txt_path = os.path.join(txt_folder, txt_filename)
                
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    lines = txt_file.readlines()
                
                if not lines:  
                    jpg_filename = os.path.splitext(txt_filename)[0] + '.jpg'
                    jpg_path = os.path.join(jpg_folder, jpg_filename)
                    
                    if os.path.exists(jpg_path):
                        os.remove(jpg_path) 
                        print(f"Usunięto plik: {jpg_path}")
                    
                    os.remove(txt_path)
                    print(f"Usunięto pusty plik: {txt_path}")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")


txt_folder_path = "SODA-D/valid_slice/labels"
jpg_folder_path = "SODA-D/valid_slice/images"


usun_txt(txt_folder_path, jpg_folder_path)
