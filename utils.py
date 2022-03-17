from config import *
from model import MultiOutputModel

def save_model(model, name):
    torch.save(model.state_dict(), os.path.join('weights', name))

def plot_history_graph(history, folder_name, name_graph):
    if not os.path.exists('graph/' + folder_name):
        os.mkdir('graph/' + folder_name)
    path = 'graph/' + folder_name
    traffic_train_acc = history['traffic_train_acc']
    color_train_acc = history['color_train_acc']
    traffic_val_acc = history['traffic_val_acc']
    color_val_acc = history['color_val_acc']

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    traffic_train_loss = history['traffic_train_loss']
    color_train_loss = history['color_train_loss']
    traffic_val_loss = history['traffic_val_loss']
    color_val_loss = history['color_val_loss']

    epochs = range(1, len(traffic_train_acc) + 1)
    plt.plot(epochs, traffic_train_acc, 'b', label='Training traffic acc')
    plt.plot(epochs, traffic_val_acc, 'r', label='Validation traffic acc')
    plt.title('Training and validation traffic accuracy')
    plt.legend()
    plt.savefig(path + '/' + name_graph[0])

    plt.figure()
    plt.plot(epochs, color_train_acc, 'b', label='Training color acc')
    plt.plot(epochs, color_val_acc, 'r', label='Validation color acc')
    plt.title('Training and validation color accuracy')
    plt.legend()
    plt.savefig(path + '/' + name_graph[1])

    plt.figure()
    plt.plot(epochs, traffic_train_loss, 'b', label='Training traffic loss')
    plt.plot(epochs, traffic_val_loss, 'r', label='Validation traffic loss')
    plt.title('Training and validation traffic loss')
    plt.legend()
    plt.savefig(path + '/' + name_graph[2])

    plt.figure()
    plt.plot(epochs, color_train_loss, 'b', label='Training color loss')
    plt.plot(epochs, color_val_loss, 'r', label='Validation color loss')
    plt.title('Training and validation color loss')
    plt.legend()
    plt.savefig(path + '/' + name_graph[3])
    plt.figure()

    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + '/' + name_graph[4])
