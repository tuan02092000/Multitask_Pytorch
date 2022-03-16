from config import *
from transforms import ImageTransform

def make_data_path_list(root_path):
    image_path = []
    traffic = []
    color = []
    for dir in os.listdir(root_path):
        folder_path = os.path.join(root_path, dir)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image_path.append(img_path)
            label = img_path.split('\\')[1].split('_')
            traffic.append(label_dict[label[0]])
            color.append(label_dict[label[1]])

    X_train, X_test = train_test_split(image_path, test_size=0.2, random_state=42)
    traffic_train, traffic_test = train_test_split(traffic, test_size=0.2, random_state=42)
    color_train, color_test = train_test_split(color, test_size=0.2, random_state=42)

    train_list = [X_train, traffic_train, color_train]
    test_list = [X_test, traffic_test, color_test]
    return train_list, test_list

class TrafficAndColorDataset(Dataset):
    def __init__(self, king_of_lists, transform, phase='train'):
        self.transform = transform
        self.phase = phase
        self.king_of_lists = king_of_lists
    def __len__(self):
        return len(self.king_of_lists[0])
    def __getitem__(self, item):
        img_path = self.king_of_lists[0][item]
        img = Image.open(img_path).convert('RGB')
        img_transform = self.transform(img, self.phase)
        traffic_label = torch.from_numpy(np.array(self.king_of_lists[1][item]))
        color_label = torch.from_numpy(np.array(self.king_of_lists[2][item]))
        return img_transform, traffic_label, color_label



if __name__ == '__main__':
    train_list, test_list = make_data_path_list('dataset')
    data_train = TrafficAndColorDataset(train_list, transform=ImageTransform(resize, mean, std), phase='train')
    data_test = TrafficAndColorDataset(test_list, transform=ImageTransform(resize, mean, std), phase='val')
    print(data_train.__getitem__(0))
    print(data_test.__getitem__(1))

