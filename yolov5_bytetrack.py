from yolox.tracker.byte_tracker import BYTETracker
from config import *
from model import *
from os import listdir
from transforms import ImageTransform
import torch.nn.functional as F

def extract_label(label_list, pred_array, top_n=1):
    pred_max = torch.topk(pred_array, top_n)[1]
    index = pred_max.cpu().numpy()
    return label_list[index.item()]

def frame_vis_generator(frame, bboxes, ids, count_frame):
    entity_id = 1
    # label_dict[entity_id][2] = 0
    # label_dict[entity_id][3] = 0

    for i, entity_id in enumerate(ids):
        x1, y1, w, h = np.round(bboxes[i]).astype(int)
        x2 = x1 + w
        y2 = y1 + h
        if (entity_id not in label_dict.keys()) or count_frame % 20 == 0:
            # convert to PIL image
            img_pil = frame[y1:y2, x1:x2]
            if img_pil.shape[0] == 0 or img_pil.shape[1] == 0:
                continue
            img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
            # img_pil = Image.fromarray(img_pil)
            img_pil = test_loader(img_pil).float()
            img_pil = Variable(img_pil, requires_grad=False)
            img_pil = img_pil.unsqueeze(0).cuda()

            # Classifier
            y_pred = model_classify(img_pil)
            prob_type = F.softmax(y_pred[0], dim=1)
            prob_color = F.softmax(y_pred[1], dim=1)
            prob_max_type = torch.max(prob_type, 1)[0].item()
            prob_max_color = torch.max(prob_color, 1)[0].item()
            type_label = extract_label(type, y_pred[0])
            color_label = extract_label(color, y_pred[1])

            label_dict[entity_id] = [type_label, color_label, prob_max_type, prob_max_color]

        text = f'{str(entity_id)}-{label_dict[entity_id][0]}-{label_dict[entity_id][1]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
    return frame

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

if __name__ == '__main__':
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Name model classifier
    # name_model = 'SqueezeNet_1_1_25_5_2'
    name_model = 'Resnet_50_test_resize'

    # Set path
    path_to_weight_model_classify = f'weights/best_model_loss_{name_model}.pth'

    # Image transform
    test_loader = transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((resize, resize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)])

    # Model classifier
    model_classify = Resnet_BackBone()
    model_classify = model_classify.to(device)
    model_classify.load_state_dict(torch.load(path_to_weight_model_classify))
    model_classify.eval()

    # Model ByteTrack
    mot_tracker = BYTETracker()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/nguyen-tuan/Documents/IVSR_LAB/weights/best_300epoch.pt')
    model.float()
    model.eval()
    video_path = "/home/nguyen-tuan/Documents/IVSR_LAB/Multitask-classification/test/download_video/traffic_11.mp4"
    vid = cv2.VideoCapture(video_path)
    ret, image_show = vid.read()
    frame_height, frame_witdth, _ = image_show.shape
    ii = 0
    out = cv2.VideoWriter(f"/home/nguyen-tuan/Documents/IVSR_LAB/Multitask-classification/test/video_test/yolo5s_bytetrack_{name_model}_traffic_11.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_witdth, frame_height))

    # Dict save label with id
    label_dict = dict()

    # Read video
    count_frame = 0
    while vid.isOpened():
        ret, frame = vid.read()
        count_frame = count_frame + 1
        if not ret:
            break
        preds = model(frame)
        # print(preds)
        # print(preds.pred[0])
        detections = preds.pred[0].cpu().numpy()
        # print(detections)
        online_targets = mot_tracker.update(detections)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 10 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
        res_img = frame_vis_generator(frame, online_tlwhs, online_ids, count_frame)
        out.write(res_img.astype(np.uint8))
    out.release()
    vid.release()
    print("Done")
   

