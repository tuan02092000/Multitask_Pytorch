from torch.utils.data import DataLoader

def get_dataloader_dict(train_dataset, test_dataset, batch_size, shuffle=True, num_worker=0):
    dataloader_dict = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_worker=num_worker),
        'val': DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_worker=num_worker)
    }
    return dataloader_dict




