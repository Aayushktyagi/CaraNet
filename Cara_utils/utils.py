import torch
import numpy as np
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))



def convert_yolo_cart(numbers, width, height):
    coord = numbers[1:]
    coord[0] = coord[0] * height
    coord[1] = coord[1] * width
    coord[2] = coord[2] * height
    coord[3] = coord[3] * width
    xmin = int(coord[0] - coord[2]/2)
    ymin = int(coord[1] - coord[3]/2)
    xmax = int(coord[0] + coord[2]/2)
    ymax = int(coord[1] + coord[3]/2)
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > 640:
        xmax = 640
    if ymax > 640:
        ymax = 640

    return xmin,ymin,xmax,ymax


def get_segment_crop(img, mask):
    mask = mask.cpu().detach().numpy()
    org_img = img
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    ret, thresh1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    img[thresh1==0] = 0
    return img



def split_image(image, split = 3):
    tiles = []
    M = image.shape[0]//split
    N = image.shape[1]//split
    for x in range(0,image.shape[0],M):
        for y in range(0, image.shape[1], N):
            tile = image[x:x+M,y:y+N]
            if tile.shape[0] == M and tile.shape[1] == N:
                tile = cv2.resize(tile, (640,640))
                tiles.append(tile)
    return tiles


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
