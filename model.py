from lib import *
from config import *

class MultiOutputModel(nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.base_model = self.model_core()
        self.x1 = nn.Linear(1000, 512)
        self.bn = nn.BatchNorm1d(512)

        # Type
        self.y1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_type)
        )
        self.y2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_color)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.bn(F.relu(self.x1(x)))
        out_type = self.y1(x)
        out_color = self.y2(x)
        return out_type, out_color
    def model_core(self):
        model = models.resnet152(pretrained=True)
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

if __name__ == '__main__':
    model = models.resnet152(pretrained=True)
    num_features = model.fc
    print(model)