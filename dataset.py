from lib import *
from config import *
from transforms import ImageTransform

def make_data_path_list(root_folder='dataset'):
    img_path = []
    type = []
    color = []

    for dir in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, dir)
        for img_name in os.listdir(folder_path):
            image = os.path.join(folder_path, img_name)
            img_path.append(image)
            label = image.split('/')[1].split('_')
            type.append(label[0])
            color.append(label[1])
    x_train, x_test = train_test_split(img_path, test_size=0.2, random_state=42)
    type_train, type_test = train_test_split(type, test_size=0.2, random_state=42)
    color_train, color_test = train_test_split(color, test_size=0.2, random_state=42)

    train_list = [x_train, type_train, color_train]
    test_list = [x_test, type_test, color_test]

    return train_list, test_list

class MyDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    def __len__(self):
        return len(self.file_list[0])
    def __getitem__(self, item):
        img_path = self.file_list[0][item]
        # print(img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img, self.phase)
        type_label = torch.from_numpy(np.array(type.index(self.file_list[1][item])))
        color_label = torch.from_numpy(np.array(color.index(self.file_list[2][item])))
        return img, type_label, color_label
if __name__ == '__main__':
    train_list, test_list = make_data_path_list('dataset_8_4')
    print('Length train list: ', len(train_list[0]))
    print('Length test list', len(test_list[0]))
    for i in range(5):
        print(f'TRAIN / Link image {i}: {train_list[0][i]}, Type: {train_list[1][i]}, Color: {train_list[2][i]}')
    for i in range(5):
        print(f'TEST / Link image {i}: {test_list[0][i]}, Type: {test_list[1][i]}, Color: {test_list[2][i]}')
    train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase='train')
    val_dataset = MyDataset(test_list, transform=ImageTransform(resize, mean, std), phase='val')
    print(train_dataset)
    print(val_dataset)
