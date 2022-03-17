from config import *
from model import MultiOutputModel
from os import listdir

model1 = MultiOutputModel()
#print(model1)
model1 = model1.to(device)
model1.load_state_dict(torch.load('resnet101.pth'))
model1.eval()

loader = transforms.Compose([transforms.Resize((resize, resize)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std)])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open('test/' + image_name)
    image = image.convert('RGB')

    image = loader(image).float()
    # print(image.shape)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU

def extract_label(label_list, pred_array, top_n=1):
    pred_max = torch.topk(pred_array, top_n)[1]
    # print(pred_array)
    out_list = []
    for i in pred_max[0]:
        out_list.append(label_list[i])
    return out_list

test = listdir('test/')  # Link to test folder

for i in test:
    image = image_loader(i)
    y_pred = model1(image)
    print('image: ', i)
    print(extract_label(traffic, y_pred[0]))
    print(extract_label(color, y_pred[1]))
    traffic_label = extract_label(traffic, y_pred[0])
    color_label = extract_label(color, y_pred[1])
    pil_im = Image.open('test/' + i)

    plt.imshow(pil_im)
    plt.xlabel(f'{traffic_label[0]} - {color_label[0]}', )
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    plt.pause(100)