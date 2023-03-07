import os
import shutil

data_path = 'Path/to/ocr_train'
folder_list = os.listdir(data_path)
print(folder_list)

for folder in folder_list:
    path = data_path + "/" + folder
    files = os.listdir(path)
    folder_len = len(files)
    train_bound = int(0.7*folder_len)
    valid_bound = int(0.9*folder_len)
    valid_data = files[train_bound:valid_bound]
    test_data = files[valid_bound:]
    valid_folder = 'Path/to/ocr_valid/'
    test_folder = 'Path/to/ocr_test/'
    os.mkdir(valid_folder + "/" + folder)
    os.mkdir(test_folder + "/" + folder)
    for img in valid_data:
        current_path = 'Path/to/ocr_train/' + folder + "/" + img
        dest_path = valid_folder + folder + "/" + img
        shutil.move(current_path, dest_path)
    for img in test_data:
        current_path = 'Path/to/ocr_train/' + folder + "/" + img
        dest_path = test_folder + folder + "/" + img
        shutil.move(current_path, dest_path)




