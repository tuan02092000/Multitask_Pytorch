from config import *
from utils import plot_history_graph, save_model
from dataloader import get_dataloader_dict
from dataset import TrafficAndColorDataset, make_data_path_list
from transforms import ImageTransform
from model import MultiOutputModel

# Model
model_1 = MultiOutputModel()
model_1 = model_1.to(device)
print(model_1)

# Loss function
criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

# Optimizer
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

# Dataset
train_list, test_list = make_data_path_list('dataset')
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

    history = {}
    history['train_loss'] = []
    history['val_loss'] = []
    history['traffic_train_loss'] = []
    history['color_train_loss'] = []
    history['traffic_val_loss'] = []
    history['color_val_loss'] = []
    history['traffic_train_acc'] = []
    history['color_train_acc'] = []
    history['traffic_val_acc'] = []
    history['color_val_acc'] = []


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
                traffic_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(traffic, 1)[1])
                color_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(color, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]
            traffic_acc = traffic_corrects.double() / dataset_sizes[phase]
            color_acc = color_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(loss)
                history['traffic_train_loss'].append(loss0)
                history['color_train_loss'].append(loss1)
                history['traffic_train_acc'].append(traffic_acc)
                history['color_train_acc'].append(color_acc)
            else:
                history['val_loss'].append(loss)
                history['traffic_val_loss'].append(loss0)
                history['color_val_loss'].append(loss1)
                history['traffic_val_acc'].append(traffic_acc)
                history['color_val_acc'].append(color_acc)

            print(
                '\n[{}] / Total_loss: {:.4f} / Traffic_loss: {:.4f} / Color_loss {:.4f}'.format(phase, loss, loss0, loss1))
            print('\n[{}] / Traffic_acc: {:.4f} / Color_acc: {:.4f} '.format(phase, traffic_acc, color_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_acc:
                print('--> Saving with loss of {}'.format(epoch_loss), 'improved over previous {}'.format(best_acc))
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Train and validation model
model_ft1, history = train_model(model_1, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
# Save model
save_model(model_ft1, 'resnet101_2.pth')

# Plot graph
plot_history_graph(history, 'resnet101', name_graph)

