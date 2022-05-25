from lib import *

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

num_epochs = 10

input_size = 64

batch_size = 64

type = ['bus', 'car', 'truck']
color = ['black', 'white', 'red', 'green', 'other']

num_type = 3
num_color = 5

lr_main = 1e-3
lr_last = 1e-4

# torch.manual_seed(1234)
# np.random.seed(1234)
# random.seed(1234)
