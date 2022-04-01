from config import *
from model import MultiOutputModel
from os import listdir

def image_loader(path_to_folder_test, image_name):
    """load image, returns cuda tensor"""
    image = Image.open(path_to_folder_test + image_name)
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

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model1 = MultiOutputModel()
    model1 = model1.to(device)
    model1.load_state_dict(torch.load('weights/best_model_color_acc_resnet152.pth'))
    model1.eval()

    loader = transforms.Compose([transforms.Resize((resize, resize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])

    path_to_folder_test = 'test/data_crop/'

    test = listdir(path_to_folder_test)  # Link to test folder
    save_img_path = 'test/save_img/'
    for i in test:
        image = image_loader(path_to_folder_test, i)
        read_img = path_to_folder_test + i

        y_pred = model1(image)
        print('image: ', i)
        print(extract_label(type, y_pred[0]))
        print(extract_label(color, y_pred[1]))
        traffic_label = extract_label(type, y_pred[0])
        color_label = extract_label(color, y_pred[1])
        # pil_im = Image.open(path_to_folder_test + i)
        # plt.imshow(pil_im)
        # plt.xlabel(f'{traffic_label[0]} - {color_label[0]}', )
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        save_img = cv2.imread(read_img)
        save_path = save_img_path + traffic_label[0] + '_' + color_label[0] + '/' + i
        cv2.imwrite(save_path, save_img)
        # plt.show()
        # plt.pause(10)