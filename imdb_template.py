# Author: Alexander Ponamarev
"""
General image database wrapper that provides a common methods for image processing
"""
import tensorflow as tf
import numpy as np
import sys, os, time

from .util import batch_iou
from .ImRead import ImRead
from .Resize import Resize
"""
from util import batch_iou
from ImRead import ImRead
from Resize import Resize
"""

class IMDB(object):

    ANCHOR_BOX = []
    INPUT_RES = None
    FEATURE_MAP_SIZE = None
    IMAGES_PATH = "path to be provided"

    def __init__(self, main_controller, resize_dim=(1024, 1024), feature_map_size=(32, 32)):

        self.samples_read_counter = 0
        self.mc = main_controller
        self.FEATURE_MAP_SIZE = feature_map_size  # width, height
        self.INPUT_RES = resize_dim  # width, height
        self.imread = ImRead(bgr2rgd_flag=True)
        self.resize = Resize(dimensinos=resize_dim)

    @property
    def samples_read_counter(self):
        return self.__samples_read_counter
    @samples_read_counter.setter
    def samples_read_counter(self, value):
        self.__samples_read_counter = value if value<self.epoch_size() else 0


    @property
    def mc(self):
        return self.__mc

    @mc.setter
    def mc(self, main_controller):
        # Let's verify all important attributes
        try:
            assert type(main_controller.OUTPUT_RES) == list, \
                "Please provide the output resolution mc.OUTPUT_RES that describes feature maps size of FCN as a list [width, height]. Will be used for bbox setup"
        except:
            raise AttributeError("mc.OUTPUT_RES was not provided. Please provide the output resolution mc.OUTPUT_RES that describes feature maps size of FCN. Will be used for bbox setup")

        try:
            assert type(main_controller.BATCH_SIZE) == int and main_controller.BATCH_SIZE > 0, "Incorrect mc.batch_size"
        except:
            raise AttributeError("mc.BATCH_SIZE was not provided.")

        try:
            assert type(main_controller.IMAGES_PATH) == str,\
                "Incorrect images path was provided. mc.IMAGES_PATH should be provided."
        except:
            raise AttributeError("mc.IMAGES_PATH was not provided.")

        self.__mc = main_controller
        self.IMAGES_PATH = main_controller.IMAGES_PATH

    @property
    def IMAGES_PATH(self):
        assert os.path.exists(self.__IMAGES_PATH), "Invalid path: {}".format(self.__IMAGES_PATH)
        return self.__IMAGES_PATH

    @IMAGES_PATH.setter
    def IMAGES_PATH(self, value):
        assert os.path.exists(value), "Invalid path: {}".format(value)
        self.__IMAGES_PATH = value

    @property
    def INPUT_RES(self):
        return self.__INPUT_RES

    @INPUT_RES.setter
    def INPUT_RES(self, value):
        """
        Sets anchor variable.

        :param value: output resolution (width, height)
        :return: self.__anchor list [x,y,h,w,b]
        """
        # assing the value
        self.__INPUT_RES = value

        # check that all the values are set
        try:
            self.ANCHOR_BOX = self.resize_anchors(IMAGE_SIZE=value,
                                                  FEATURE_MAP_SIZE=self.FEATURE_MAP_SIZE)  # x,y,h,w,b
        except AttributeError:
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    @property
    def FEATURE_MAP_SIZE(self):
        return self.__FEATURE_MAP_SIZE

    @FEATURE_MAP_SIZE.setter
    def FEATURE_MAP_SIZE(self, value):
        """
        Sets anchor variable.

        :param value: output resolution (width, height)
        :return: self.__anchor list [x,y,h,w,b]
        """
        # assing the value
        self.__FEATURE_MAP_SIZE = value

        # check that all the values are set
        try:
            self.ANCHOR_BOX = self.resize_anchors(IMAGE_SIZE=self.INPUT_RES,
                                                  FEATURE_MAP_SIZE=value)  # x,y,h,w,b
        except AttributeError:
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def resize_anchors(self, IMAGE_SIZE, FEATURE_MAP_SIZE):
        """
        Sets anchor variable.

        :param value: output resolution (width, height)
        :return: anchor list [x,y,h,w,b]
        """
        H, W, B = FEATURE_MAP_SIZE[0], FEATURE_MAP_SIZE[1], 9
        IMAGE_WIDTH, IMAGE_HEIGHT = [v for v in IMAGE_SIZE]

        anchor_shapes = np.reshape(
            [np.array(
                [[36., 37.], [366., 174.], [115., 59.],
                 [162., 87.], [38., 90.], [258., 173.],
                 [224., 108.], [78., 170.], [72., 43.]])] * H * W,
            (H, W, B, 2)
        )
        center_x = np.reshape(
            np.transpose(
                np.reshape(
                    np.array([np.arange(1, W + 1) * float(IMAGE_WIDTH) / (W + 1)] * H * B),
                    (B, H, W)
                ),
                (1, 2, 0)
            ),
            (H, W, B, 1)
        )
        center_y = np.reshape(
            np.transpose(
                np.reshape(
                    np.array([np.arange(1, H + 1) * float(IMAGE_HEIGHT) / (H + 1)] * W * B),
                    (B, W, H)
                ),
                (2, 1, 0)
            ),
            (H, W, B, 1)
        )
        anchors = np.reshape(
            np.concatenate((center_x, center_y, anchor_shapes), axis=3),
            (-1, 4)
        )

        return anchors # x,y,h,w,b

    def find_anchor_ids(self, img_boxes):
        """
        Identifies anchor ids_per_img responsible for object detection
        :param img_boxes:
        :return: anchor ids_per_img
        """
        ids_per_img = []
        id_iterator = set()
        aid = len(self.ANCHOR_BOX)
        for box in img_boxes:
            overlaps = batch_iou(self.ANCHOR_BOX, box)
            for id in np.argsort(overlaps)[::-1]:
                if overlaps[id]<=0:
                    break
                if id not in id_iterator:
                    id_iterator.add(id)
                    aid = id
                    break
            ids_per_img.append(aid)
        return ids_per_img

    def estimate_deltas(self, bboxes, anchor_ids):
        """Calculates the deltas of ANCHOR_BOX and ground truth boxes.
        :param bboxes: an array of ground trueth bounding boxes (bboxes) for an image [center_x, center_y,
        width, height]
        :param anchor_ids: ids per each ground truth box that have the highest IOU
        :return: [anchor_center_x_delta,anchor_center_y_delta, log(anchor_width_scale), log(anchor_height_scale)]
        """
        assert len(bboxes)==len(anchor_ids),\
            "Incorrect arrays provided for bboxes (len[{}]) and aids (len[{}]).".format(len(bboxes), len(anchor_ids)) +\
            " Provided arrays should have the same length. "
        delta_per_img = []
        for box, aid in zip(bboxes, anchor_ids):
            # calculate deltas
            # unpack the box
            box_cx, box_cy, box_w, box_h = box
            # initialize a delta array [x,y,w,h]
            if not(box_w > 0) or not(box_h > 0):
                raise ValueError("Incorrect bbox size: height {}, width {}".format(box_h, box_w))
            delta = [0] * 4
            delta[0] = (box_cx - self.ANCHOR_BOX[aid][0]) / box_w
            delta[1] = (box_cy - self.ANCHOR_BOX[aid][1]) / box_h
            delta[2] = np.log(box_w / self.ANCHOR_BOX[aid][2])
            delta[3] = np.log(box_h / self.ANCHOR_BOX[aid][3])
            delta_per_img.append(delta)
        return delta_per_img

    def __provide_img_file_name(self, id):
        """
        Protocol describing the implementation of a method that provides the name of the image file based on
        an image id. Should be implemented for each of the datasets separately
        :param id: dataset specific image id
        :return: string containing file name
        """
        raise NotImplementedError

    def __provide_img_tags(self, id):
        """
        Protocol describing the implementation of a method that provides tags for the image file based on
        an image id. Should be implemented for each of the datasets separately
        :param id: dataset specific image id
        :return: an array containing the list of tags
        """
        raise NotImplementedError

    def __provide_img_id(self, id):
        """
        Protocol describing the implementation of a method that provides image id
        for the dataset. This image id will be used in all subsequent methods.
        Should be implemented for each of the datasets separately
        :param id: id of the array
        :return: an id within image available image ids.
        """
        raise NotImplementedError

    def epoch_size(self):
        """
        Protocol describing the implementation of a method that provides the size of the epoch.
        Should be implemented for each of the datasets separately
        :return: int - the size of the epoch
        """
        raise NotImplementedError



