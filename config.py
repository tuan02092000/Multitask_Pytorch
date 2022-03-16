from lib import *

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

num_epochs = 10

input_size = 64

num_workers = 0
batch_size = 64
val_ratio = 0.2

traffic = ['bus', 'car', 'truck']
color = ['black', 'white', 'red', 'green']

num_traffic = 3
num_color = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lrlast = .001
lrmain = .0001

label_dict = {
    'bus': [1, 0, 0],
    'car': [0, 1, 0],
    'truck': [0, 0, 1],
    'black': [1, 0, 0, 0],
    'white': [0, 1, 0, 0],
    'red': [0, 0, 1, 0],
    'green': [0, 0, 0, 1]
}
