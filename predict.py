from config import *
from model import *
from os import listdir
from transforms import ImageTransform

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

def predict_and_save_type_and_color(path_to_folder_test, path_to_save_img):
    test = listdir(path_to_folder_test)  # Link to test folder
    for i in test:
        image = image_loader(path_to_folder_test, i)
        read_img = path_to_folder_test + i

        y_pred = model1(image)
        # print('image: ', i)
        # print(extract_label(type, y_pred[0]))
        # print(extract_label(color, y_pred[1]))
        traffic_label = extract_label(type, y_pred[0])
        color_label = extract_label(color, y_pred[1])
        # print(read_img)
        save_img = cv2.imread(read_img)
        save_path = path_to_save_img + '/' + traffic_label[0] + '_' + color_label[0] + '/' + i

        cv2.imwrite(save_path, save_img)

def predict_and_save_type(path_to_folder_test, path_to_save_img):
    test = listdir(path_to_folder_test)  # Link to test folder
    for i in test:
        image = image_loader(path_to_folder_test, i)
        read_img = path_to_folder_test + i

        y_pred = model1(image)

        traffic_label = extract_label(type, y_pred[0])
        color_label = extract_label(color, y_pred[1])

        save_img = cv2.imread(read_img)
        save_path = path_to_save_img + '/' + traffic_label[0] + '/' + i
        cv2.imwrite(save_path, save_img)


def predict_and_save_color(path_to_folder_test, path_to_save_img):
    test = listdir(path_to_folder_test)  # Link to test folder
    for i in test:
        image = image_loader(path_to_folder_test, i)
        read_img = path_to_folder_test + i

        y_pred = model1(image)

        traffic_label = extract_label(type, y_pred[0])
        color_label = extract_label(color, y_pred[1])

        save_img = cv2.imread(read_img)
        save_path = path_to_save_img + '/' + color_label[0] + '/' + i
        cv2.imwrite(save_path, save_img)

def predict_and_show_type_and_color(path_to_folder_test):
    test = listdir(path_to_folder_test)  # Link to test folder
    for i in test:
        image = image_loader(path_to_folder_test, i)

        y_pred = model1(image)

        traffic_label = extract_label(type, y_pred[0])
        color_label = extract_label(color, y_pred[1])

        pil_im = Image.open(path_to_folder_test + i)
        plt.imshow(pil_im)
        plt.xlabel(f'{traffic_label[0]} - {color_label[0]}', )
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.pause(10)
        plt.show()

if __name__ == '__main__':
    # Choose model
    # name_model = 'Densenet_161'
    # name_model = 'Resnet_152'
    # name_model = 'SqueezeNet1_1'
    name_model = 'Squeezenet_1_1_fix_finetune'
    label_type_color = ['bus_black', 'bus_green', 'bus_other', 'bus_red', 'bus_white',
                        'car_black', 'car_green', 'car_other', 'car_red', 'car_white',
                        'truck_black', 'truck_green', 'truck_other', 'truck_red', 'truck_white']

    # config link
    # path_to_folder_test = 'test/img_test/'
    path_to_folder_test = 'test/tests/'
    path_to_weight_model = f'weights/best_model_loss_{name_model}.pth'

    # 1. type and color
    path_to_save_img_type_color = f'test/predict_model_select/{name_model}/predict_save_img_type_color'
    # 2. type
    path_to_save_img_type = f'test/predict_model_select/{name_model}/predict_save_img_type'
    # 3. color
    path_to_save_img_color = f'test/predict_model_select/{name_model}/predict_save_img_color'

    if not os.path.exists(f'test/predict_model_select/{name_model}'):
        os.mkdir(f'test/predict_model_select/{name_model}')
    if not os.path.exists(path_to_save_img_type_color):
        os.mkdir(path_to_save_img_type_color)
    for label in label_type_color:
        if not os.path.exists(f'{path_to_save_img_type_color}/{label}'):
            os.mkdir(f'{path_to_save_img_type_color}/{label}')


    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    model1 = SqueezeNet_BackBone()
    model1 = model1.to(device)
    model1.load_state_dict(torch.load(path_to_weight_model))
    model1.eval()

    # Image transform
    loader = transforms.Compose([transforms.Resize((resize, resize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])

    # Predict and show type and color
    predict_and_show_type_and_color(path_to_folder_test)

    # Predict and save type and color
    # predict_and_save_type_and_color(path_to_folder_test, path_to_save_img_type_color)

    # Predict and save type
    # predict_and_save_type(path_to_folder_test, path_to_save_img_type)

    # Predict and save color
    # predict_and_save_color(path_to_folder_test, path_to_save_img_color)
