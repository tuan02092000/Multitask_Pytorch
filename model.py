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
        model_ft = models.resnet152(pretrained=True)  # Choose your model backbone
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
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_type, bias=True)
        )
        self.y2 = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_color, bias=True)
        )

    def forward(self, x):
        x = self.base_model(x)
        x1 = self.avgpool(x)
        x1 = x1.reshape(x1.size(0), -1)
        out_type = self.y1(x1)
        out_color = self.y2(x1)

        return out_type, out_color
    def model_core(self):
        model_ft = models.mobilenet_v3_small(pretrained=True).features  # Choose your model backbone
        for param in model_ft.parameters():
            param.requires_grad = False
        return model_ft

class Resnet_BackBone(nn.Module):
    def __init__(self):
        super(Resnet_BackBone, self).__init__()

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
        model_ft = models.resnet50(pretrained=True)  # Choose your model backbone
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 512)
        return model_ft

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

if __name__ == '__main__':
    model = models.densenet161()
    print(model)
