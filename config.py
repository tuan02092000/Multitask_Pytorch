from lib import *

# resize = 224
resize = 128
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

num_epochs = 15

input_size = 64

batch_size = 32

type = ['bus', 'car', 'truck']
color = ['black', 'white', 'red', 'green', 'other']

num_type = 3
num_color = 5

lr_main = 1e-3
lr_last = 1e-4

color_dict = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'other': (255, 255, 0)
}

NAME_MP4 = 'traffic_48'

# torch.manual_seed(1234)
# np.random.seed(1234)
# random.seed(1234)
