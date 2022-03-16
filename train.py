from config import *
from dataloader import get_dataloader_dict
from dataset import TrafficAndColorDataset, make_data_path_list
from transforms import ImageTransform

# Model
model_ft = models.resnet50(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
# print(model_ft)

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

model_1 = multi_output_model(model_ft)
model_1 = model_1.to(device)
# print(model_1)

criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

optim = optim.Adam(
    [
        {"params": model_1.resnet_model.parameters()},
        {"params": model_1.x1.parameters(), "lr": lrlast},
        {"params": model_1.y1o.parameters(), "lr": lrlast},
        {"params": model_1.y2o.parameters(), "lr": lrlast},
    ],
    lr=lrmain)

# optim = optim.Adam(model_1.parameters(),lr=lrmain)#, momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


train_list, test_list = make_data_path_list('dataset')  # Link to dataset folder
data_train = TrafficAndColorDataset(train_list, transform=ImageTransform(resize, mean, std), phase='train')
data_test = TrafficAndColorDataset(test_list, transform=ImageTransform(resize, mean, std), phase='val')
dataloaders_dict = get_dataloader_dict(train_dataset=data_train, test_dataset=data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

dataset_sizes = {
    'train': len(train_list[0]),
    'val': len(test_list[0])
}

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            traffic_corrects = 0.0
            color_corrects = 0.0

            # Iterate over data.
            for inputs, traffic, color in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)

                traffic = traffic.to(device)
                color = color.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs)
                    outputs = model(inputs)

                    loss0 = criterion[0](outputs[0], torch.max(traffic.float(), 1)[1])
                    loss1 = criterion[1](outputs[1], torch.max(color.float(), 1)[1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = loss0 + loss1
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print(torch.max(outputs[0], 1)[1],torch.max(gen, 1)[1],torch.max(outputs[0], 1)[1]==torch.max(gen, 1)[1])
                traffic_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(traffic, 1)[1])
                color_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(color, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]
            traffic_acc = traffic_corrects.double() / dataset_sizes[phase]
            color_acc = color_corrects.double() / dataset_sizes[phase]

            print(
                '{} Total_loss: {:.4f} / Traffic_loss: {:.4f} / Color_loss {:.4f}'.format(
                    phase, loss, loss0, loss1))
            print('{} Traffic_acc: {:.4f} / Color_acc: {:.4f} '.format(
                phase, traffic_acc, color_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_acc:
                print('--> Saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_acc))
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft1 = train_model(model_1, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
torch.save(model_ft1.state_dict(), 'resnet50.pth')

