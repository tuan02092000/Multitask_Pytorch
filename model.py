import torch

from lib import *
from config import *

class MultiOutputModel(nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.base_model = self.model_core()
        self.x1 = nn.Linear(1000, 512)
        # Type
        self.y1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.2, inplace=True),
            nn.ReLU(),
            nn.Linear(256, num_type)
        )
        self.y2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.2, inplace=True),
            nn.ReLU(),
            nn.Linear(256, num_color),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.x1(x))
        # out_type = self.y1(x)
        # out_color = self.y2(x)

        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model = models.efficientnet_b0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return model
class MultiOutputModel_3(nn.Module):
    def __init__(self):
        super(MultiOutputModel_3, self).__init__()
        self.base_model = self.model_core()

        self.y1 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1000, num_type)
            )
        self.y2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1000, num_color)
        )

    def forward(self, x):
        x = self.base_model(x)
        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model = models.squeezenet1_1(pretrained=True).features
        for param in model.parameters():
            param.requires_grad = False
        return model

class MultiOutputModel_DenseNet(nn.Module):
    def __init__(self):
        super(MultiOutputModel_DenseNet, self).__init__()
        self.base_model = self.model_core()

        self.y1 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=num_type, bias=True)
            )
        self.y2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=num_color, bias=True)
        )

    def forward(self, x):
        x = self.base_model(x)
        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model = models.densenet201(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return model

class MultiOutputModel_2(torch.nn.Module):
    def __init__(self):
        super(MultiOutputModel_2, self).__init__()

        self.base_model = self.model_core()

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=1e-2)
        # self.x2 =  nn.Linear(128,64)
        # nn.init.xavier_normal_(self.x2.weight)
        # self.x3 =  nn.Linear(64,32)
        # nn.init.xavier_normal_(self.x3.weight)
        # comp head 1

        # heads
        self.y1 = nn.Linear(256, num_type)
        nn.init.xavier_normal_(self.y1.weight)  #
        self.y2 = nn.Linear(256, num_color)
        nn.init.xavier_normal_(self.y2.weight)
        # self.sm = nn.Softmax(dim=1)
        # self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x = self.base_model(x)
        x1 = self.bn1(F.relu(self.x1(x)))

        # heads
        y1o = self.y1(x1)
        y2o = self.y2(x1)

        return y1o, y2o
    def model_core(self):
        model_ft = models.resnet152(pretrained=False)  # Choose your model backbone
        for param in model_ft.parameters():
            param.requires_grad = False
        # Fine tuning
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 512)
        return model_ft

class MobileNetV3_BackBone(nn.Module):
    def __init__(self):
        super(MobileNetV3_BackBone, self).__init__()

        self.base_model = self.model_core()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.y1 = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_type, bias=True)
        )
        self.y2 = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_color, bias=True)
        )

    def forward(self, x):
        x = self.base_model(x)
        x1 = self.avgpool(x)
        # x1 = x1.reshape(x1.size(0), -1)
        x1 = torch.flatten(x1, 1)
        out_type = self.y1(x1)
        out_color = self.y2(x1)

        return out_type, out_color
    def model_core(self):
        model_ft = models.mobilenet_v3_small(pretrained=False).features  # Choose your model backbone
        for param in model_ft.parameters():
            param.requires_grad = False
        return model_ft

# class Resnet_BackBone(nn.Module):
#     def __init__(self):
#         super(Resnet_BackBone, self).__init__()
#
#         self.base_model = self.model_core()
#
#         self.y1 = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(in_features=512, out_features=num_type, bias=True),
#         )
#         self.y2 = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(in_features=512, out_features=num_color, bias=True),
#         )
#
#     def forward(self, x):
#         x = self.base_model(x)
#         x = torch.flatten(x, 1)
#         out_type = self.y1(x)
#         out_color = self.y2(x)
#         return out_type, out_color
#     def model_core(self):
#         model_ft = models.resnet50(pretrained=True)  # Choose your model backbone
#         # for param in model_ft.parameters():
#         #     param.requires_grad = False
#         model_wo_fc = nn.Sequential(*(list(model_ft.children())[:-1]))
#         # num_ftrs = model_ft.fc.in_features
#         # model_ft.fc = nn.Linear(num_ftrs, 512)
#         return model_wo_fc

class Resnet_BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.model_wo_fc = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.y1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_type)
        )
        self.y2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_color)
        )
    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)
        type = self.y1(x)
        color = self.y2(x)
        return type, color


class SqueezeNet_BackBone(nn.Module):
    def __init__(self):
        super(SqueezeNet_BackBone, self).__init__()
        self.base_model = self.model_core()

        self.y1 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1000, num_type)
            # nn.Linear(1000, 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.25),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(256, num_type)
            )

        self.y2 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1000, num_color)
            # nn.Linear(1000, 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.25),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(256, num_color)
            )

    def forward(self, x):
        x = self.base_model(x)
        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model = models.squeezenet1_1(pretrained=True).features
        for param in model.parameters():
            param.requires_grad = False
        return model

class Densenet_BackBone(nn.Module):
    def __init__(self):
        super(Densenet_BackBone, self).__init__()

        self.base_model = self.model_core()

        self.y1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_type, bias=True),
        )
        self.y2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_color, bias=True),
        )

    def forward(self, x):
        x = self.base_model(x)
        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model_ft = models.densenet161(pretrained=True)  # Choose your model backbone
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 512)
        return model_ft

class Shufflenet_BackBone(nn.Module):
    def __init__(self):
        super(Shufflenet_BackBone, self).__init__()
        self.base_model = self.model_core()
        self.y1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 3)
        )
        self.y2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 5)
        )
    def forward(self, x):
        x = self.base_model(x)
        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model = models.shufflenet_v2_x0_5(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        return model

class VGG_BackBone(nn.Module):
    def __init__(self):
        super(VGG_BackBone, self).__init__()

        self.base_model = self.model_core()

        self.y1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_type, bias=True),
        )
        self.y2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_color, bias=True),
        )

    def forward(self, x):
        x = self.base_model(x)
        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model_ft = models.vgg19(pretrained=True)  # Choose your model backbone
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 512)
        return model_ft

if __name__ == '__main__':
    model = models.resnet34()
    print(model)
    # for name, child in model.named_children():
    #     print(name)
    #     # print(child)
    # print(model)
    # resnet = models.resnet34(pretrained=True)
    # print(list(resnet.children())[-3:])
    # model_wo_fc = nn.Sequential(*(list(resnet.children())[:-1]))
    # print(model_wo_fc)

