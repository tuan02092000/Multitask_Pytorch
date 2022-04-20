from lib import *
from config import *
from dataloader import get_dataloader_dict
from dataset import MyDataset, make_data_path_list
from model import *
from utils import *
from transforms import ImageTransform

def train(model, dataloader_dict : dict, dataset_size : dict, criterion, optimzier, scheduler, name_model, num_epochs=num_epochs):
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    since = time.time()
    best_model_loss_wts = copy.deepcopy(model.state_dict())
    best_model_type_acc_wts = copy.deepcopy(model.state_dict())
    best_model_color_acc_wts = copy.deepcopy(model.state_dict())
    best_model_acc_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0.0
    best_acc_type = 0.0
    best_acc_color = 0.0

    writer_loss = SummaryWriter(f'runs/{name_model}/Loss')
    writer_acc = SummaryWriter(f'runs/{name_model}/Acc')
    step = 0

    for epoch in range(num_epochs):
        model.to(device)
        # torch.backends.cudnn.benchmark = True
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0.0
            type_correct = 0.0
            color_correct = 0.0

            for inputs, type_label, color_label in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                type_label = type_label.type(torch.LongTensor).to(device)
                color_label = color_label.type(torch.LongTensor).to(device)

                # zero the parameter gradients
                optimzier.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds_type = torch.max(outputs[0], 1)
                    _, preds_color = torch.max(outputs[1], 1)

                    loss0 = criterion[0](outputs[0], type_label)
                    loss1 = criterion[1](outputs[1], color_label)
                    loss = loss0 + loss1

                    # Backward + optimizer
                    if phase == 'train':
                        loss.backward()
                        optimzier.step()

                if epoch != 0:
                    if phase == 'train':
                        writer_loss.add_scalar('Training traffic loss: ', loss0, global_step=step)
                        writer_loss.add_scalar('Training color loss: ', loss1, global_step=step)
                        writer_acc.add_scalar('Training type acc: ', type_acc, global_step=step)
                        writer_acc.add_scalar('Training color acc: ', color_acc, global_step=step)
                        writer_acc.add_scalar('Training type and color acc', epoch_acc, global_step=step)
                    else:
                        writer_loss.add_scalar('Val traffic loss: ', loss0, global_step=step)
                        writer_loss.add_scalar('Val color loss: ', loss1, global_step=step)
                        writer_acc.add_scalar('Val type acc: ', type_acc, global_step=step)
                        writer_acc.add_scalar('Val color acc: ', color_acc, global_step=step)
                        writer_acc.add_scalar('Val type and color acc', epoch_acc, global_step=step)
                    step += 1


                running_loss += loss.item() * inputs.size(0)
                type_correct += torch.sum(preds_type == type_label.data)
                # print(preds_type == type_label.data)
                color_correct += torch.sum(preds_color == color_label.data)
                # print(preds_color == color_label.data)
                for i in range(batch_size):
                    running_correct += torch.sum((preds_type[i] == type_label[i].data) and (preds_color[i] == color_label[i].data))
                # print(running_correct)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_correct.double() / dataset_size[phase]
            type_acc = type_correct.double() / dataset_size[phase]
            color_acc = color_correct.double() / dataset_size[phase]

            str = '{} / Epoch {}/{} / Traffic_loss: {:.4f} / Color_loss: {:.4f} / Epoch_acc: {:.4f} / Traffic_acc: {:.4f} / Color_acc: {:.4f}\n'.format(
                phase, epoch, num_epochs - 1, loss0, loss1, epoch_acc, type_acc, color_acc)
            if phase == 'train':
                save_to_txt(f'{name_model}', 'train', 'train.txt', str)
            else:
                save_to_txt(f'{name_model}', 'val', 'val.txt', str)

            print(
                '\n[{}] / Total_loss: {:.4f} / Traffic_loss: {:.4f} / Color_loss {:.4f}'.format(phase, loss, loss0, loss1))
            print('\n[{}] / Epoch_acc: {:.4f} / Traffic_acc: {:.4f} / Color_acc: {:.4f} '.format(phase, epoch_acc, type_acc, color_acc))

            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print('\n--> Saving with total loss of {}'.format(epoch_loss), 'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_loss_wts = copy.deepcopy(model.state_dict())
                save_model(best_model_loss_wts, f'best_model_loss_{name_model}.pth')
            if phase == 'val' and type_acc > best_acc_type:
                print('\n--> Saving with acc of type: {}'.format(type_acc), 'improved over previous {}'.format(best_acc_type))
                best_acc_type = type_acc
                best_model_type_acc_wts = copy.deepcopy(model.state_dict())
                save_model(best_model_type_acc_wts, f'best_model_type_acc_{name_model}.pth')
            if phase == 'val' and color_acc > best_acc_color:
                print('\n--> Saving with acc of color: {}'.format(color_acc), 'improved over previous {}'.format(best_acc_color))
                best_acc_color = color_acc
                best_model_color_acc_wts = copy.deepcopy(model.state_dict())
                save_model(best_model_color_acc_wts, f'best_model_color_acc_{name_model}.pth')
            if phase == 'val' and epoch_acc > best_acc:
                print('\n--> Saving with acc: {}'.format(epoch_acc), 'improved over previous {}'.format(best_acc))
                best_acc = epoch_acc
                best_model_acc_wts = copy.deepcopy(model.state_dict())
                save_model(best_model_acc_wts, f'best_model_acc_{name_model}.pth')

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best total loss: {:.4f}'.format(best_loss))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    # model_loss.load_state_dict(best_model_wts)
    # return model

if __name__ == '__main__':
    # Dataset
    train_list, test_list = make_data_path_list('dataset_8_4')

    train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase='train')
    val_dataset = MyDataset(test_list, transform=ImageTransform(resize, mean, std), phase='val')

    # Dataloader
    dataloader_dict = get_dataloader_dict(train_dataset=train_dataset, test_dataset=val_dataset, batch_size=batch_size)

    dataset_size = {
        'train': len(train_list[0]),
        'val': len(test_list[0])
    }

    # Model
    model = Densenet_BackBone()

    # Loss function
    criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    # Optimizer
    optimizer = optim.Adam(
        [
            {"params": model.base_model.parameters()},
            # {"params": model.avgpool.parameters(), "lr": lr_last},
            {"params": model.y1.parameters(), "lr": lr_last},
            {"params": model.y2.parameters(), "lr": lr_last},
        ],
        lr=lr_main)
    optimizer_ft = optimizer
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training
    train(model, dataloader_dict, dataset_size, criterion, optimizer_ft, exp_lr_scheduler, 'Densenet', num_epochs)

