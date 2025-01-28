import os

def usun_9(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            filtered_lines = [line for line in lines if not line.startswith('9')]
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(filtered_lines)
            
            print(f"Przetworzono plik: {filename}")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

folder_path = "SODA-D/foo/labels"

usun_9(folder_path)
