"""
COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic
"""
from __future__ import print_function
import os, time, threading
import numpy as np
from random import shuffle
from pycocotools.coco import COCO
from imdb_template import imdb_template as IMDB
from util import visualization, convertToFixedSize, sparse_to_dense
# Syntax: class(object) create a class inheriting from an object to allow new stype variable management


class coco(IMDB):
    imgIds = []
    catIds = []
    @property
    def imgIds(self):
        return self.__imgIds
    @imgIds.setter
    def imgIds(self, values):
        assert type(values)==list,\
            "imgIds is incorrect. Array is expected. {} was recieved.".format(type(values).__name__)
        shuffle_flag = self.shuffle
        if shuffle_flag:
            shuffle(values)

        self.__imgIds = values

    name = None
    @property
    def name(self):
        assert type(self.__name) == str, \
            "Coco dataset name was not set correctly. Coco dataset should be initialized with a name: coco(name='train', path, mc)"""
        return self.__name
    @name.setter
    def name(self, value):
        assert type(value) == str, \
            "Coco dataset name was not set correctly. Coco dataset should be initialized with a name: coco(name='train', path, mc)"""
        self.__name = "coco_" + value

    annotations_file = "File name to be provided"
    @property
    def annotations_file(self):
        assert os.path.exists(self.__annotations_file),\
            "Invalid path was provided for the data set. The following path doesn't exist: {}"\
            .format(self.__annotations_file)
        return self.__annotations_file
    @annotations_file.setter
    def annotations_file(self, value):
        assert os.path.exists(value), "Annotations file doesn't exis at path {}. Please provide a full path to annotatinos." \
            .format(value)
        self.__annotations_file = value

    BATCH_CLASSES = None
    @property
    def BATCH_CLASSES(self):
        return self.__BATCH_CLASSES

    @BATCH_CLASSES.setter
    def BATCH_CLASSES(self, values):
        for value in values:
            assert value in self.CLASS_NAMES_AVAILABLE,\
                "BATCH_CLASSES array is incorrect. The following class (from the array) " + \
                "is not present in self.CLASS_NAMES_AVAILABLE array: {}.".\
                format(value)
        self.__BATCH_CLASSES = values

        # build an array of file indicies
        img_ids = []
        cat_ids = []
        for class_name in self.__BATCH_CLASSES:
            cat_id = self.coco.getCatIds(catNms=[class_name])
            cat_ids.extend(cat_id)
            img_ids.extend(self.coco.getImgIds(catIds=cat_id))

        # update image ids
        self.imgIds = img_ids
        self.catIds = cat_ids

    def __init__(self, coco_name, main_controller, shuffle=True, resize_dim=(1024, 1024), prefetched_batches=5):
        """
        COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic.
        The dataset should be initialized with:
        :param name: string with a name of the dataset. Good names can be 'train', 'test', or 'va', but I am not a stickler =)
        :param path: a full path to the folder containing images and annotations folder
        :param mc: main controller containing remaining parameters necessary for the proper initialization. mc should contain:
            - mc.batch_size - an integer greater than 0
            - mc.ANNOTATIONS_FILE_NAME - a file name located in the coco_path/annotations
            - mc.BATCH_CLASSES - an array of classes to be learned (at least 1)
        """

        IMDB.__init__(self, resize_dim=resize_dim,
                      feature_map_size=main_controller.OUTPUT_RES,
                      main_controller=main_controller,
                      prefetched_batches=prefetched_batches)

        self.name = coco_name
        self.shuffle = shuffle

        #1. Get an array of image indicies
        assert type(main_controller.ANNOTATIONS_FILE_NAME)==str,\
            "Provide a name of the file containing annotations in mc.ANNOTATIONS_FILE_NAME"
        self.annotations_file = main_controller.ANNOTATIONS_FILE_NAME
        self.coco = COCO(self.annotations_file)
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.CLASS_NAMES_AVAILABLE = [category['name'] for category in categories]
        self.CATEGORIES = set([category['supercategory'] for category in categories])
        assert type(main_controller.BATCH_CLASSES) == list and len(main_controller.BATCH_CLASSES)>0,\
            "Provide a list of classes to be learned in this batch through mc.BATCH_CLASSES"
        self.BATCH_CLASSES = main_controller.BATCH_CLASSES

    def provide_img_id(self, id):
        return self.imgIds[id]

    def provide_epoch_size(self):
        return len(self.imgIds)

    def provide_img_file_name(self, id):
        """
        Protocol describing the implementation of a method that provides the name of the image file based on
        an image id.
        :param id: dataset specific image id
        :return: string containing file name
        """

        descriptions = self.coco.loadImgs(id)[0]
        return descriptions['file_name']

    def provide_img_tags(self, id, coco_labels=False):
        """
        Protocol describing the implementation of a method that provides tags for the image file based on
        an image id.
        :param id: dataset specific image id
        :param coco_labels(optional): indicates wherher coco label ids should be returned
        default value is False - BATCH_CLASS ids should be returned
        :return: an array containing the list of tags
        """


        # Extract annotation ids
        ann_ids = self.coco.getAnnIds(imgIds=[id], catIds=self.catIds, iscrowd=None)
        # get all annotations available
        anns = self.coco.loadAnns(ids=ann_ids)

        # parse annotations into a list
        cat_ids = []
        for ann in anns:
            if ann['iscrowd']!=1:
                cat_ids.append(ann['category_id'])

        if coco_labels:
            return cat_ids
        else:
            cats = self.coco.loadCats(ids=cat_ids)
            cat_ids = [self.BATCH_CLASSES.index(cat['name']) for cat in cats]
        return cat_ids

    def provide_img_gtbboxes(self, id, resize=True):
        """
        Protocol describing the implementation of a method that provides ground truth bounding boxes
        for the image file based on an image id.
        :param id: dataset specific image id
        :return: an array containing the list of bounding boxes with the following format
        [center_x, center_y, width, height]
        """
        bboxes = []

        # Extract annotation ids
        ann_ids = self.coco.getAnnIds(imgIds=[id],
                                      catIds=self.coco.getCatIds(catNms=self.BATCH_CLASSES)
                                      )
        # get all annotations available
        anns = self.coco.loadAnns(ids=ann_ids)
        # parse annotations into a list
        for ann in anns:
            bbox = self.resize.bboxResize(ann['bbox']) if resize else ann['bbox']
            bboxes.append([bbox[0] + bbox[2] / 2,
                           bbox[1] + bbox[3] / 2,
                           bbox[2],
                           bbox[3]
                           ])
        return bboxes

    def transform_cocoID2batchID(self, ids):
        anns = self.coco.loadCats(ids=ids)
        label_names = [ann['name'] for ann in anns]
        batch_class_ids = [self.BATCH_CLASSES.index(v) for v in label_names]
        return batch_class_ids

    def read_batch(self, step):

        image_per_batch, \
        sparse_label_per_batch, \
        sparse_gtbox_per_batch, \
        sparse_aids_per_batch, \
        sparse_deltas_per_batch = super(coco, self).read_batch(step)

        # Convert into a flattened out list
        label_indices, \
        bbox_indices, \
        box_delta_values, \
        mask_indices, \
        box_values = convertToFixedSize(aidx_per_batch=sparse_aids_per_batch,
                                        label_per_batch=sparse_label_per_batch,
                                        box_delta_per_batch=sparse_deltas_per_batch,
                                        bbox_per_batch=sparse_gtbox_per_batch)

        # Extract variables to make it more readable
        batch = self.mc.BATCH_SIZE
        n_anchors = len(self.ANCHOR_BOX)
        n_classes = len(self.BATCH_CLASSES)
        n_labels = len(label_indices)

        # Dense boxes
        label_indices = sparse_to_dense(label_indices, [batch, n_anchors, n_classes], np.ones(n_labels, dtype=np.float))
        bbox_deltas = sparse_to_dense(bbox_indices, [batch, n_anchors, 4], box_delta_values)
        mask = np.reshape(sparse_to_dense(mask_indices, [batch, n_anchors], np.ones(n_labels, dtype=np.float)),
                          [batch, n_anchors, 1])
        box_values = sparse_to_dense(bbox_indices, [batch, n_anchors, 4], box_values)

        return {'imgs': np.array(image_per_batch, dtype=np.float),
                'dense_labels': label_indices,
                'masks': mask,
                'bbox_deltas': bbox_deltas,
                'bbox_values': box_values}

    def enqueue_batch(self, sess):

        try:
            self.__batch_drawn += 1
        except:
            self.__batch_drawn = 0

        input_dict = self.read_batch(self.__batch_drawn)
        input_list = [input_dict['imgs'],
                      input_dict['bbox_deltas'],
                      input_dict['masks'],
                      input_dict['dense_labels'],
                      input_dict['bbox_values']]

        super(coco, self).enqueue_batch(input_list, sess)



if __name__ == "__main__":

    import tensorflow as tf
    from easydict import EasyDict as edict

    mc = edict()
    mc.BATCH_SIZE = 10
    # Dimensions for:
    # imgs, bbox_deltas, masks, dense_labels, bbox_values
    mc.OUTPUT_SHAPES = [[768, 768, 3], [32**2*9, 4], [32**2*9,1], [32**2*9, 3], [32**2*9, 4]]
    mc.OUTPUT_DTYPES = [tf.float32, tf.int32, tf.int32, tf.int32, tf.float32]
    mc.ANNOTATIONS_FILE_NAME = '/Users/aponamaryov/Downloads/coco_train_2014/annotations/instances_train2014.json'
    mc.BATCH_CLASSES = ['person', 'car', 'bicycle']
    mc.OUTPUT_RES = (32, 32)
    mc.IMAGES_PATH = '/Users/aponamaryov/Downloads/coco_train_2014/images'
    c = coco(coco_name="train", main_controller=mc, resize_dim=(768, 768), prefetched_batches=10)

    print("The name of the dataset: {}".format(c.name))
    print("Batch provides images for:  \n", c.BATCH_CLASSES)

    print("\nQueue 4 batches...")
    with tf.Session().as_default() as sess:

        start_timer = time.time()
        for i in range(4):
            enqueue_thread = threading.Thread(target=c.enqueue_batch, args=[sess])
            enqueue_thread.isDaemon()
            enqueue_thread.start()
        end_timer = time.time()

        print("It took {:.3f} seconds to enqueue 4 batches asynchronously.".format(end_timer-start_timer))
        print("For comparison, let's queue in the same 4 batches sequentially.")
        start_timer = time.time()
        for _ in range(4): c.enqueue_batch(sess)
        end_timer = time.time()
        print("It took {:.3f} seconds to enqueue 4 batches sequentially.".format(end_timer - start_timer))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        imgs, bbox_deltas, masks, dense_labels, bbox_values = sess.run(c.get_batch)

        # shutdown everything to avoid zombies
        sess.run(c.queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        sess.close()

        print("Images have the following shape:", imgs.shape)
        print("bbox_deltas have the following shape:", bbox_deltas.shape)
        print("masks have the following shape:", masks.shape)
        print("dense_labels have the following shape:", dense_labels.shape)
        print("bbox_values have the following shape:", bbox_values.shape)

    print(len(c.ANCHOR_BOX))
    c.BATCH_CLASSES = ['person', 'dog', 'cat', 'car']
    print(c.BATCH_CLASSES)
    print(len(c.imgIds))