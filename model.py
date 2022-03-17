from config import *

########## Model
# model_ft = models.resnet50(pretrained=True)  # resnet50
# model_ft = models.resnet101(pretrained=True)
# for param in model_ft.parameters():
#     param.requires_grad = False
# print(model_ft)
#
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 512)


# model_ft = models.vgg16(pretrained=True)  # vgg16
# for param in model_ft.parameters():
#     param.requires_grad = False
# print(model_ft)
#
# num_ftrs = model_ft.classifier[6].in_features
# model_ft.classifier[6] = nn.Linear(num_ftrs, 512)

class MultiOutputModel(torch.nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()

        self.resnet_model = self.model_core()

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
    def model_core(self):
        model_ft = models.resnet101(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        # Fine tuning
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 512)
        return model_ft

if __name__ == '__main__':
    model_1 = MultiOutputModel()
    model_1 = model_1.to(device)
    print(model_1)