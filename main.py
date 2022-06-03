import numpy as np
import os
import torch
import pandas as pd
from utils import write_pd

# Calculate precision and recall

def insert_to_dict_type_color(path_to_folder_label, path_to_folder_model_predict):
        label_test = dict()
        label_predict = dict()
        for folder in os.listdir(path_to_folder_label):
                label_test[folder] = []
                for img in os.listdir(os.path.join(path_to_folder_label, folder)):
                        label_test[folder].append(img)

        for folder in os.listdir(path_to_folder_model_predict):
                label_predict[folder] = []
                for img in os.listdir(os.path.join(path_to_folder_model_predict, folder)):
                        label_predict[folder].append(img)
        return label_test, label_predict

def insert_to_dict(path_to_folder_label, path_to_folder_model_predict):
        label_test_type_color = dict()
        label_predict_type_color = dict()

        label_test_type = {'bus': [], 'car': [], 'truck': []}
        label_predict_type = {'bus': [], 'car': [], 'truck': []}

        label_test_color = {'black': [], 'white': [], 'red': [], 'green': [], 'other': []}
        label_predict_color = {'black': [], 'white': [], 'red': [], 'green': [], 'other': []}

        for folder in os.listdir(path_to_folder_label):
                type, color = folder.split('_')
                label_test_type_color[folder] = []

                for img in os.listdir(os.path.join(path_to_folder_label, folder)):
                        label_test_type_color[folder].append(img)
                        label_test_type[type].append(img)
                        label_test_color[color].append(img)

        for folder in os.listdir(path_to_folder_model_predict):
                type, color = folder.split('_')
                label_predict_type_color[folder] = []

                for img in os.listdir(os.path.join(path_to_folder_model_predict, folder)):
                        label_predict_type_color[folder].append(img)
                        label_predict_type[type].append(img)
                        label_predict_color[color].append(img)

        return label_test_type_color, label_predict_type_color, label_test_type, label_predict_type, label_test_color, label_predict_color

def count_TP_FN_FP_type_color(label_test, label_predict):
        N = len(label_type_color)
        count_TP = np.zeros(N)
        count_FN = np.zeros(N)
        count_FP = np.zeros(N)
        for index, lb in enumerate(label_type_color):
                for img in label_test[lb]:
                        if img in label_predict[lb]:
                                count_TP[index] += 1
                        else:
                                count_FN[index] += 1
                for img in label_predict[lb]:
                        if img not in label_test[lb]:
                                count_FP[index] += 1
        return count_TP, count_FN, count_FP
def count_TP_FN_FP_type(label_test, label_predict):
        N = len(label_type)
        count_TP = np.zeros(N)
        count_FN = np.zeros(N)
        count_FP = np.zeros(N)
        for index, lb in enumerate(label_type):
                for img in label_test[lb]:
                        if img in label_predict[lb]:
                                count_TP[index] += 1
                        else:
                                count_FN[index] += 1
                for img in label_predict[lb]:
                        if img not in label_test[lb]:
                                count_FP[index] += 1
        return count_TP, count_FN, count_FP
def count_TP_FN_FP_color(label_test, label_predict):
        N = len(label_color)
        count_TP = np.zeros(N)
        count_FN = np.zeros(N)
        count_FP = np.zeros(N)
        for index, lb in enumerate(label_color):
                for img in label_test[lb]:
                        if img in label_predict[lb]:
                                count_TP[index] += 1
                        else:
                                count_FN[index] += 1
                for img in label_predict[lb]:
                        if img not in label_test[lb]:
                                count_FP[index] += 1
        return count_TP, count_FN, count_FP

def cal_precision_recall(count_TP, count_FN, count_FP):
        # N = len(label_type_color)
        # precision_type_color = np.zeros(len(label_type_color))
        # recall_type_color = np.zeros(len(label_type_color))
        # for i in range(N):
        #         precision_type_color[i] = count_TP[i] / (count_TP[i] + count_FP[i])
        #         recall_type_color[i] = count_TP[i] / (count_TP[i] + count_FN[i])
        precision = count_TP / (count_TP + count_FP)
        recall = count_TP / (count_TP + count_FN)
        return precision, recall
def cal_F1_Score(precision, recall):
        return 2 * (precision * recall) / (precision + recall)
if __name__ == '__main__':
        # Choose model
        # name_model = 'Densenet_161'
        # name_model = 'Resnet_152'
        # name_model = 'SqueezeNet1_1'
        name_model = 'Resnet_50'

        # Choose link
        path_to_folder_label = './test/my_test/'
        path_to_folder_model_predict = f'./test/predict_model_select/{name_model}/predict_save_img_type_color'

        label_type_color = ['bus_black', 'bus_green', 'bus_other', 'bus_red', 'bus_white',
                            'car_black', 'car_green', 'car_other', 'car_red', 'car_white',
                            'truck_black', 'truck_green', 'truck_other', 'truck_red', 'truck_white']
        label_type = ['bus', 'truck', 'car']
        label_color = ['black', 'white', 'red', 'green', 'other']

        label_test, label_predict, label_test_type, label_predict_type, label_test_color, label_predict_color = insert_to_dict(path_to_folder_label, path_to_folder_model_predict)

        # Calculate precision and recall of 'Type and Color'
        count_TP, count_FN, count_FP = count_TP_FN_FP_type_color(label_test, label_predict)
        precision_type_color, recall_type_color = cal_precision_recall(count_TP, count_FN, count_FP)
        f1_score_type_color = cal_F1_Score(precision_type_color, recall_type_color)

        # Calculate precision and recall of 'Type'
        count_TP_type, count_FN_type, count_FP_type = count_TP_FN_FP_type(label_test_type, label_predict_type)
        precision_type, recall_type = cal_precision_recall(count_TP_type, count_FN_type, count_FP_type)
        f1_score_type = cal_F1_Score(precision_type, recall_type)

        # Calculate precision and recall of 'Color'
        count_TP_color, count_FN_color, count_FP_color = count_TP_FN_FP_color(label_test_color, label_predict_color)
        precision_color, recall_color = cal_precision_recall(count_TP_color, count_FN_color, count_FP_color)
        f1_score_color = cal_F1_Score(precision_color, recall_color)


        data_type_color = {'Label': label_type_color,
                           'Precision': precision_type_color,
                           'Recall': recall_type_color,
                           'F1 Score': f1_score_type_color}
        data_type = {'Label': label_type,
                     'Precision': precision_type,
                     'Recall': recall_type,
                     'F1 Score': f1_score_type}
        data_color = {'Label': label_color,
                      'Precision': precision_color,
                      'Recall': recall_color,
                      'F1 Score': f1_score_color}

        df_type_color = pd.DataFrame(data_type_color)
        df_type = pd.DataFrame(data_type)
        df_color = pd.DataFrame(data_color)

        str = [str(df_type_color), str(df_type), str(df_color)]
        write_pd(name_model, str)

        # print type and color
        # df_precision = pd.DataFrame(precision_type_color, label_type_color)
        # df_recall = pd.DataFrame(recall_type_color, label_type_color)
        # df_f1Score = pd.DataFrame(f1_score_type_color, label_type_color)

        # print type
        # df_precision_type = pd.DataFrame(precision_type, label_type)
        # df_recall_type = pd.DataFrame(recall_type, label_type)
        # df_f1Score_type = pd.DataFrame(f1_score_type, label_type)

        # print color
        # df_precision_color = pd.DataFrame(precision_color, label_color)
        # df_recall_color = pd.DataFrame(recall_color, label_color)
        # df_f1Score_color = pd.DataFrame(f1_score_color, label_color)

        # print('\nPRECISION - Type and Color: ')
        # print(df_precision)
        # print('\nRECALL - Type and Color')
        # print(df_recall)
        # print('\nF1 Score - Type and Color')
        # print(df_f1Score)

        # print('\nPRECISION - Type')
        # print(df_precision_type)
        # print('\nRECALL - Type')
        # print(df_recall_type)
        # print('\nF1 Score - Type')
        # print(df_f1Score_type)

        # print('\nPRECISION - Color')
        # print(df_precision_color)
        # print('\nRECALL - Color')
        # print(df_recall_color)
        # print('\nF1 Score - Color')
        # print(df_f1Score_color)



