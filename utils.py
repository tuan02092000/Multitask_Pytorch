from lib import *
from config import *

def save_model(model, name):
    if not os.path.exists('./weights'):
        os.mkdir('weights')
    torch.save(model, os.path.join('weights', name))

def save_to_txt(folder_name, phase, file_name, str):
    path_folder = './note/' + folder_name
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    path_folder_phase = path_folder + '/' + phase
    if not os.path.exists(path_folder_phase):
        os.mkdir(path_folder_phase)
    root_path = path_folder_phase + '/' + file_name
    with open(root_path, 'a', encoding='utf-8') as f:
        f.writelines(str)
