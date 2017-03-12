"""
COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic
"""
import os
import numpy as np
from random import shuffle
if __name__=='__main__':
    from PythonAPI.pycocotools.coco import COCO
    from imdb_template import IMDB
    from util import convertToFixedSize, sparse_to_dense
else:
    from .PythonAPI.pycocotools.coco import COCO
    from .imdb_template import IMDB
    from .util import convertToFixedSize, sparse_to_dense
# Syntax: class(object) create a class inheriting from an object to allow new stype variable management


class coco(IMDB):

    def __init__(self, coco_name, main_controller, shuffle=True, resize_dim=(1024, 1024)):
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
        self.shuffle = shuffle


        self.imgIds = []
        self.catIds = []
        self.name = coco_name

        IMDB.__init__(self, resize_dim=resize_dim,
                      feature_map_size=main_controller.OUTPUT_RES,
                      main_controller=main_controller)

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

    def __provide_img_id(self, id):
        return self.imgIds[id]

    def epoch_size(self):
        return len(self.imgIds)

    def __provide_img_file_name(self, id):
        """
        Protocol describing the implementation of a method that provides the name of the image file based on
        an image id.
        :param id: dataset specific image id
        :return: string containing file name
        """

        descriptions = self.coco.loadImgs(id)[0]
        return descriptions['file_name']


    def __provide_img_tags(self, id, coco_labels=True):
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
        bbox_values = []
        segmentation = []
        for ann in anns:
            if ann['iscrowd']!=1:
                cat_ids.append(ann['category_id'])
                bbox_values.append(ann['bbox'])
                segmentation.append(ann['segmentation'])

        if not(coco_labels):
            cats = self.coco.loadCats(ids=cat_ids)
            cat_ids = [self.BATCH_CLASSES.index(cat['name']) for cat in cats]

        return cat_ids, bbox_values, segmentation

    def get_sample(self):

        while True:
            # 1. Get img_id
            img_id = self.__provide_img_id(self.samples_read_counter)

            # 2. Get annotatinos
            labels, gtbboxes, segmentation = self.__provide_img_tags(img_id)

            self.samples_read_counter+=1

            # 2. Read the file name
            file_name = self.__provide_img_file_name(img_id)
            file_path = os.path.join(self.IMAGES_PATH, file_name)
            try:
                im = self.resize.imResize(self.imread.read(file_path))
                gtbboxes = [self.resize.bboxResize(gtbbox) for gtbbox in gtbboxes]
                # provide anchor ids for each image
                aids = self.find_anchor_ids(gtbboxes)
                # calculate deltas for each anchor and add them to the delta_per_batch
                deltas = self.estimate_deltas(gtbboxes, aids)
                break
            except:
                pass

        dense_anns = self.__map_to_grid(labels, gtbboxes, aids, deltas)
        """
        im = tf.convert_to_tensor(im, dtype=tf.float32, name='image')
        labels = tf.convert_to_tensor(dense_anns['dense_labels'], dtype=tf.int32, name='labels')
        aids = tf.convert_to_tensor(dense_anns['masks'], dtype=tf.int32, name='aids')
        deltas = tf.convert_to_tensor(dense_anns['bbox_deltas'], dtype=tf.float32, name='bbox_deltas')
        bbox_values = tf.convert_to_tensor(dense_anns['bbox_values'], dtype=tf.float32, name='bbox_values')
        """
        im = im
        labels = dense_anns['dense_labels']
        aids = dense_anns['masks']
        deltas = dense_anns['bbox_deltas']
        bbox_values = dense_anns['bbox_values']

        return [im, labels, aids, deltas, bbox_values]


    def __map_to_grid(self, sparse_label, sparse_gtbox, sparse_aids, sparse_deltas):

        # Convert into a flattened out list
        label_indices, \
        bbox_indices, \
        box_delta_values, \
        mask_indices, \
        box_values = convertToFixedSize(aidx=sparse_aids,
                                        labels=sparse_label,
                                        boxes_deltas=sparse_deltas,
                                        bboxes=sparse_gtbox)

        # Extract variables to make it more readable
        n_anchors = len(self.ANCHOR_BOX)
        n_classes = len(self.CLASS_NAMES_AVAILABLE) #len(self.BATCH_CLASSES)
        n_labels = len(label_indices)

        # Dense boxes
        label_indices = sparse_to_dense(label_indices, [n_anchors, n_classes], np.ones(n_labels, dtype=np.float))
        bbox_deltas = sparse_to_dense(bbox_indices, [n_anchors, 4], box_delta_values)
        mask = np.reshape(sparse_to_dense(mask_indices, [n_anchors], np.ones(n_labels, dtype=np.float)),
                          [n_anchors, 1])
        box_values = sparse_to_dense(bbox_indices, [n_anchors, 4], box_values)

        return {'dense_labels': label_indices,
                'masks': mask,
                'bbox_deltas': bbox_deltas,
                'bbox_values': box_values}


if __name__=="__main__":
    import tensorflow as tf
    import time, threading
    from easydict import EasyDict as edict

    mc = edict()
    mc.BATCH_SIZE = 10
    # Dimensions for:
    mc.ANNOTATIONS_FILE_NAME = '/Users/aponamaryov/Downloads/coco_train_2014/annotations/instances_train2014.json'
    mc.BATCH_CLASSES = ['person', 'car', 'bicycle']
    mc.OUTPUT_RES = (32, 32)
    mc.IMAGES_PATH = '/Users/aponamaryov/Downloads/coco_train_2014/images'


    c = coco(coco_name="train", main_controller=mc, resize_dim=(768, 768))



    with tf.Session().as_default() as sess:

        placeholder_im = tf.placeholder(dtype=tf.float32,
                                        shape=[c.resize.dimension_targets[0],c.resize.dimension_targets[0],3],
                                        name="img")
        placeholder_labels = tf.placeholder(dtype=tf.int32,
                                            shape=[len(c.ANCHOR_BOX), len(c.CLASS_NAMES_AVAILABLE)],
                                            name='labels')
        placeholder_aids = tf.placeholder(dtype=tf.int32,
                                          shape=[len(c.ANCHOR_BOX),1],
                                          name='anchor_ids')
        placeholder_deltas = tf.placeholder(dtype=tf.float32,
                                            shape=[len(c.ANCHOR_BOX),4],
                                            name='anchor_deltas')
        placeholder_bbox_values = tf.placeholder(dtype=tf.float32,
                                                 shape=[len(c.ANCHOR_BOX),4],
                                                 name='bbox_values')

        capacity = mc.BATCH_SIZE*10

        q = tf.FIFOQueue(capacity=mc.BATCH_SIZE*10,
                         dtypes=[tf.float32, tf.int32, tf.int32, tf.float32, tf.float32])

        enqueue_op = q.enqueue([placeholder_im, placeholder_labels, placeholder_aids, placeholder_deltas, placeholder_bbox_values])
        dequeue_op = q.dequeue()

        def fill_q():
            while capacity>sess.run(q.size()):
                im, labels, aids, deltas, bbox_values = c.get_sample()
                feed_dict = {placeholder_im:im,
                             placeholder_labels:labels,
                             placeholder_aids:aids,
                             placeholder_deltas:deltas,
                             placeholder_bbox_values:bbox_values}
                sess.run(enqueue_op, feed_dict=feed_dict)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        fill_q()

        print("Queue has {} elements before dequeuing.".format(sess.run(q.size())))
        cur_img, cur_labels, cur_aids, cur_deltas, cur_bbox_values = sess.run(dequeue_op)
        print(cur_img.shape, cur_labels.shape, cur_aids.shape, cur_deltas.shape, cur_bbox_values.shape)
        print("After dequeuing the queue has {} elements left.".format(sess.run(q.size())))

        print("Queue shapes: {}".format(q.shapes))

        # shutdown everything to avoid zombies
        coord.request_stop()
        coord.join(threads)
        sess.run(q.close(cancel_pending_enqueues=True))
