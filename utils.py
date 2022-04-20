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
def write_pd(model_name, str):
    if not os.path.exists('./note'):
        os.mkdir('note')
    path_to_model = './note/' + model_name
    if not os.path.exists(path_to_model):
        os.mkdir(path_to_model)
    path_to_pre_recall = path_to_model + '/' + 'precision_recall'
    if not os.path.exists(path_to_pre_recall):
        os.mkdir(path_to_pre_recall)
    type_folder = ['type_color', 'type', 'color']
    for index, type in enumerate(type_folder):
        path_to_folder = path_to_pre_recall + '/' + type
        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)
        path_to_save = path_to_folder + '/' + f'{model_name}_{type}.txt'
        with open(path_to_save, mode='w') as f:
            f.write(str[index])