import cv2
import torch
import numpy as np


def preprocess(img):
    im = cv2.resize(img, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0)
    return data


def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1, 2, 0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8')
    inp = inp[:, :, ::-1]
    return inp
