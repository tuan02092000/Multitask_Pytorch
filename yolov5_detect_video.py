from config import *
from model import *
from os import listdir
from transforms import ImageTransform

def extract_label(label_list, pred_array, top_n=1):
    pred_max = torch.topk(pred_array, top_n)[1]
    index = pred_max.cpu().numpy()
    return label_list[index.item()]

if __name__ == '__main__':
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Name model classifier
    name_model = 'SqueezeNet1_1'

    # Set path
    path_to_weight_model_classify = f'weights/best_model_loss_{name_model}.pth'

    # Image transform
    test_loader = transforms.Compose([transforms.Resize((resize, resize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])

    # Model classifier
    model_classify = SqueezeNet_BackBone()
    model_classify = model_classify.to(device)
    model_classify.load_state_dict(torch.load(path_to_weight_model_classify))
    model_classify.eval()


    # Model
    model_predict = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/nguyen-tuan/Documents/IVSR_LAB/weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom

    # # Images
    video_path = '/home/nguyen-tuan/Documents/IVSR_LAB/Multitask-classification/test/video_test/test.webm'  
    cap = cv2.VideoCapture(video_path)
    
    if cap.isOpened() == False:
        print("Error opening video")
        exit()
    
    while cap.isOpened():
        ret, frame = cap.read()
        results = model_predict(frame)
        if ret:
            for i in range(results.xyxy[0].shape[0]):
                coord = results.xyxy[0][i].cpu().numpy()
                x1 = int(coord[0].item())
                y1 = int(coord[1].item())
                x2 = int(coord[2].item())
                y2 = int(coord[3].item())
                crop_img = frame[y1:y2, x1:x2]
                
                img_pil = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_pil)
                img_pil = test_loader(img_pil).float()
                img_pil = Variable(img_pil, requires_grad=False)
                img_pil = img_pil.unsqueeze(0).cuda()

                y_pred = model_classify(img_pil)
                type_label = extract_label(type, y_pred[0])
                color_label = extract_label(color, y_pred[1])
                cv2.putText(frame, f'{type_label} - {color_label}', (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                cv2.imshow('Frame',frame)
                if cv2.waitKey(1)== ord('q'):
                    break
        else:
            break  
    cap.release()
    cv2.destroyAllWindows()     

