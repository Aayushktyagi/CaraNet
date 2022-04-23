import torch
from utils import intersection_over_union

def load_model(model_path, device):
    #load model
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path)
    detection_model.eval()
    if train_on_gpu:
        detection_model.to(device)

    return detection_model

def model_prediction(model, image):
#     with model.no_grad():
    results = model(image, size=640)
    return results



def get_detection_data(model, tiles,bbox, model):
    # get image and corresponding masks
    iou_loss_list = []
    for i, tile in enumerate(tiles):
        results = model_prediction(model, tile)
        preds = results.xyxy[0]
        gt = bbox[i]
        preds = preds.cpu().detach().numpy()
        #compute loss
        n_i = len(gt)
        n_j = len(preds)
        iou_mat = np.empty((n_i, n_j))
        for i in range(n_i):
            for j in range(n_j):
                iou_mat[i, j] = intersection_over_union(gt[i], preds[j])
        if iou_mat.size > 0:
            iou = np.max(iou_mat, axis = 1)
            iou_sum = np.mean(iou)
#             iou_sum = iou.sum()/ np.count_nonzero(iou)
            iou_loss = 1 - iou_sum
            iou_loss_list.append(iou_loss)
    if len(iou_loss_list) != 0:
        return (sum(iou_loss_list)/ len(iou_loss_list))
    else:
        return 0.001
