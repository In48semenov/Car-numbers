import cv2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..utils.mtcnn_util import *


class PNet(nn.Module):
    def __init__(self, is_train=False):

        super(PNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 10, 3, 1)),
                    ("prelu1", nn.PReLU(10)),
                    ("pool1", nn.MaxPool2d((2, 5), ceil_mode=True)),
                    ("conv2", nn.Conv2d(10, 16, (3, 5), 1)),
                    ("prelu2", nn.PReLU(16)),
                    ("conv3", nn.Conv2d(16, 32, (3, 5), 1)),
                    ("prelu3", nn.PReLU(32)),
                ]
            )
        )

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class ONet(nn.Module):
    def __init__(self, is_train=False):

        super(ONet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1)),
                    ("prelu1", nn.PReLU(32)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(32, 64, 3, 1)),
                    ("prelu2", nn.PReLU(64)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(64, 64, 3, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("pool3", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv4", nn.Conv2d(64, 128, 1, 1)),
                    ("prelu4", nn.PReLU(128)),
                    ("flatten", Flatten()),
                    ("conv5", nn.Linear(1280, 256)),
                    ("drop5", nn.Dropout(0.25)),
                    ("prelu5", nn.PReLU(256)),
                ]
            )
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a


class MtcnnNet:
    def __init__(self, device, p_model_path=None, o_model_path=None):
        self.device = device
        self.pnet = None
        if p_model_path is not None:
            self.pnet = PNet().to(device)
            self.pnet.load_state_dict(
                torch.load(p_model_path, map_location=lambda storage, loc: storage)
            )
            self.pnet.eval()

        self.onet = None
        if o_model_path is not None:
            self.onet = ONet().to(device)
            self.onet.load_state_dict(
                torch.load(o_model_path, map_location=lambda storage, loc: storage)
            )
            self.onet.eval()

    def predict(self, image, mini_lp_size=(50, 15)):
        bboxes = np.array([])
        if self.pnet is not None:
            bboxes = self.detect_pnet(image, mini_lp_size)
        if self.onet is not None:
            bboxes = self.detect_onet(image, bboxes)
        return bboxes

    def detect_pnet(self, image, min_lp_size):
        # start = time.time()

        thresholds = 0.6  # lp detection thresholds
        nms_thresholds = 0.7

        # BUILD AN IMAGE PYRAMID
        height, width, channel = image.shape
        min_height, min_width = height, width

        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        factor_count = 0
        while min_height > min_lp_size[1] and min_width > min_lp_size[0]:
            scales.append(factor**factor_count)
            min_height *= factor
            min_width *= factor
            factor_count += 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for scale in scales:
                sw, sh = math.ceil(width * scale), math.ceil(height * scale)
                img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
                img = torch.FloatTensor(preprocess(img)).to(self.device)
                offset, prob = self.pnet(img)
                probs = prob.cpu().data.numpy()[
                    0, 1, :, :
                ]  # probs: probability of a face at each sliding window
                offsets = (
                    offset.cpu().data.numpy()
                )  # offsets: transformations to true bounding boxes
                # applying P-Net is equivalent, in some sense, to moving 12x12 window with stride 2
                stride, cell_size = (2, 5), (12, 44)
                # indices of boxes where there is probably a lp
                # returns a tuple with an array of row idx's, and an array of col idx's:
                inds = np.where(probs > thresholds)

                if inds[0].size == 0:
                    boxes = None
                else:
                    # transformations of bounding boxes
                    tx1, ty1, tx2, ty2 = [
                        offsets[0, i, inds[0], inds[1]] for i in range(4)
                    ]
                    offsets = np.array([tx1, ty1, tx2, ty2])
                    score = probs[inds[0], inds[1]]
                    # P-Net is applied to scaled images
                    # so we need to rescale bounding boxes back
                    bounding_box = np.vstack(
                        [
                            np.round((stride[1] * inds[1] + 1.0) / scale),
                            np.round((stride[0] * inds[0] + 1.0) / scale),
                            np.round(
                                (stride[1] * inds[1] + 1.0 + cell_size[1]) / scale
                            ),
                            np.round(
                                (stride[0] * inds[0] + 1.0 + cell_size[0]) / scale
                            ),
                            score,
                            offsets,
                        ]
                    )
                    boxes = bounding_box.T
                    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
                    boxes[keep]

                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]

            if bounding_boxes != []:
                bounding_boxes = np.vstack(bounding_boxes)
                keep = nms(bounding_boxes[:, 0:5], nms_thresholds)
                bounding_boxes = bounding_boxes[keep]
            else:
                bounding_boxes = np.zeros((1, 9))
            # use offsets predicted by pnet to transform bounding boxes
            bboxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5],  x1, y1, x2, y2, score

            bboxes[:, 0:4] = np.round(bboxes[:, 0:4])

            # print("pnet predicted in {:2.3f} seconds".format(time.time() - start))

            return bboxes

    def detect_onet(self, image, bboxes):
        size = (94, 24)
        thresholds = 0.8  # face detection thresholds
        nms_thresholds = 0.7
        height, width, channel = image.shape

        num_boxes = len(bboxes)
        [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

        img_boxes = np.zeros((num_boxes, 3, size[1], size[0]))

        for i in range(num_boxes):
            img_box = np.zeros((h[i], w[i], 3))

            img_box[dy[i]: (edy[i] + 1), dx[i]: (edx[i] + 1), :] = image[
                y[i]: (ey[i] + 1), x[i]: (ex[i] + 1), :
            ]

            # resize
            img_box = cv2.resize(img_box, size, interpolation=cv2.INTER_LINEAR)

            img_boxes[i, :, :, :] = preprocess(img_box)

        img_boxes = torch.FloatTensor(img_boxes).to(self.device)
        offset, prob = self.onet(img_boxes)
        offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
        probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds)[0]
        bboxes = bboxes[keep]
        bboxes[:, 4] = probs[keep, 1].reshape((-1,))  # assign score from stage 2
        offsets = offsets[keep]

        bboxes = calibrate_box(bboxes, offsets)
        keep = nms(bboxes, nms_thresholds, mode="min")
        bboxes = bboxes[keep]
        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])

        return bboxes
