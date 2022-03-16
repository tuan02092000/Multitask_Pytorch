from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet50(pretrained=True)
#for param in model_ft.parameters():
#    param.requires_grad = False
#print(model_ft)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 512)

class multi_output_model(torch.nn.Module):
    def __init__(self, model_core):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=1e-2)
        # self.x2 =  nn.Linear(128,64)
        # nn.init.xavier_normal_(self.x2.weight)
        # self.x3 =  nn.Linear(64,32)
        # nn.init.xavier_normal_(self.x3.weight)
        # comp head 1

        # heads
        self.y1o = nn.Linear(256, num_traffic)
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(256, num_color)
        nn.init.xavier_normal_(self.y2o.weight)

        # self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x = self.resnet_model(x)
        x1 = self.bn1(F.relu(self.x1(x)))

        # heads
        y1o = F.softmax(self.y1o(x1), dim=1)
        y2o = F.softmax(self.y2o(x1), dim=1)  # should be sigmoid

        return y1o, y2o

model1 = multi_output_model(model_ft)
#print(model1)
model1 = model1.to(device)
model1.load_state_dict(torch.load('resnet50.pth'))
model1.eval()

loader = transforms.Compose([transforms.Resize((resize, resize)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std)])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open('test/' + image_name)
    image = image.convert('RGB')

    image = loader(image).float()
    print(image.shape)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU

def extract_label(label_list, pred_array, top_n=1):
    pred_max = torch.topk(pred_array, top_n)[1]
    print(pred_array)
    out_list = []
    for i in pred_max[0]:
        out_list.append(label_list[i])
    return out_list


from os import listdir
from matplotlib.pyplot import imshow


test = listdir('test/') # Link to test folder

for i in test:
    image = image_loader(i)
    y_pred = model1(image)
    print(i)
    print('')
    print('image: ', i)
    print(extract_label(traffic, y_pred[0]))
    print(extract_label(color, y_pred[1]))
    pil_im = Image.open('test/' + i)

    plt.imshow(np.asarray(pil_im))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()