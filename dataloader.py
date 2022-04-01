from lib import *

def get_dataloader_dict(train_dataset, test_dataset, batch_size, num_workers=0):
    dataloader_dict = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        'val': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    }
    return dataloader_dict
