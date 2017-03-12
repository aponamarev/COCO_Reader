# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Utility functions."""

import numpy as np
import time
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

def iou(box1, box2):
  """Compute the Intersection-Over-Union of two given boxes.

  Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
  Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
  """

  lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
      max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
  if lr > 0:
    tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
      intersection = tb*lr
      union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

      return intersection/union

  return 0

def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.

  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union

def nms(boxes, probs, threshold):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

# TODO(bichen): this is not equivalent with full NMS. Need to improve it.
def recursive_nms(boxes, probs, threshold, form='center'):
  """Recursive Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format) or [xmin, ymin, xmax, ymax]
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  if form == 'center':
    # convert to diagonal format
    boxes = np.array([bbox_transform(b) for b in boxes])

  areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
  hidx = boxes[:, 0].argsort()
  keep = [True]*len(hidx)

  def _nms(hidx):
    order = probs[hidx].argsort()[::-1]

    for idx in range(len(order)):
      if not keep[hidx[order[idx]]]:
        continue
      xx2 = boxes[hidx[order[idx]], 2]
      for jdx in range(idx+1, len(order)):
        if not keep[hidx[order[jdx]]]:
          continue
        xx1 = boxes[hidx[order[jdx]], 0]
        if xx2 < xx1:
          break
        w = xx2 - xx1
        yy1 = max(boxes[hidx[order[idx]], 1], boxes[hidx[order[jdx]], 1])
        yy2 = min(boxes[hidx[order[idx]], 3], boxes[hidx[order[jdx]], 3])
        if yy2 <= yy1:
          continue
        h = yy2-yy1
        inter = w*h
        iou = inter/(areas[hidx[order[idx]]]+areas[hidx[order[jdx]]]-inter)
        if iou > threshold:
          keep[hidx[order[jdx]]] = False

  def _recur(hidx):
    if len(hidx) <= 20:
      _nms(hidx)
    else:
      mid = len(hidx)/2
      _recur(hidx[:mid])
      _recur(hidx[mid:])
      _nms([idx for idx in hidx if keep[idx]])

  _recur(hidx)

  return keep

def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
  """Build a dense matrix from sparse representations.

  Args:
    sp_indices: A [0-2]-D array that contains the index to place values.
    shape: shape of the dense matrix.
    values: A {0,1}-D array where values corresponds to the index in each row of
    sp_indices.
    default_value: values to set for indices not specified in sp_indices.
  Return:
    A dense numpy N-D array with shape output_shape.
  """

  assert len(sp_indices) == len(values), \
      'Length of sp_indices is not equal to length of values'

  array = np.ones(output_shape) * default_value
  for idx, value in zip(sp_indices, values):
    array[tuple(idx)] = value
  return array

def bgr_to_rgb(ims):
  """Convert a list of images from BGR format to RGB format."""
  out = []
  for im in ims:
    out.append(im[:,:,::-1])
  return out

def bbox_transform(bbox):
  """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform') as scope:
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = cx-w/2
    out_box[1] = cy-h/2
    out_box[2] = cx+w/2
    out_box[3] = cy+h/2

  return out_box

def bbox_transform_inv(bbox):
  """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform_inv') as scope:
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]]*4

    width       = xmax - xmin + 1.0
    height      = ymax - ymin + 1.0
    out_box[0]  = xmin + 0.5*width 
    out_box[1]  = ymin + 0.5*height
    out_box[2]  = width
    out_box[3]  = height

  return out_box

class Timer(object):
  def __init__(self):
    self.total_time   = 0.0
    self.calls        = 0
    self.start_time   = 0.0
    self.duration     = 0.0
    self.average_time = 0.0

  def tic(self):
    self.start_time = time.time()

  def toc(self, average=True):
    self.duration = time.time() - self.start_time
    self.total_time += self.duration
    self.calls += 1
    self.average_time = self.total_time/self.calls
    if average:
      return self.average_time
    else:
      return self.duration

def safe_exp(w, thresh):
  """Safe exponential function for tensors."""

  slope = np.exp(thresh)
  with tf.variable_scope('safe_exponential'):
    lin_region = tf.to_float(w > thresh)

    lin_out = slope*(w - thresh + 1.)
    exp_out = tf.exp(w)

    out = lin_region*lin_out + (1.-lin_region)*exp_out
  return out
def convertToFixedSize(aidx, labels, boxes_deltas, bboxes):
    """Convert a 2d arrays of inconsistent size (varies based on n of objests) into
    a list of tuples or triples to keep the consistent dimensionality across all
    images (invariant to the number of objects)"""

    label_indices = []
    bbox_indices = []
    box_delta_values = []
    mask_indices = []
    box_values = []

    # Initialize a tracker of unique [img_ids, anchor] tuples and counter of labels
    aidx_set = set()
    label_counter = 0
    num_discarded_labels = 0

    for lbl_num in range(len(labels)):
        label_counter += 1
        # To keep a track of added label/ANCHOR_BOX create a list of ANCHOR_BOX
        # (for each image [i]) corresponding to objects
        ojb_anchor_id = aidx[lbl_num]
        obj_label = labels[lbl_num]
        box_deltas = boxes_deltas[lbl_num]
        box_xyhw = bboxes[lbl_num]
        if (ojb_anchor_id) not in aidx_set:
            aidx_set.add(ojb_anchor_id)
            # 2. Create a list of unique objects in the batch through triples [im_index, anchor, label]
            label_indices.append([ojb_anchor_id, obj_label])
            mask_indices.append([ojb_anchor_id])
            # For bounding boxes duplicate [im_num, anchor_id] 4 times (one time of each coordinates x,y,w,h
            bbox_indices.extend([[ojb_anchor_id, xywh] for xywh in range(4)])
            box_delta_values.extend(box_deltas)
            box_values.extend(box_xyhw)
        else:
            num_discarded_labels += 1
    return label_indices, bbox_indices, box_delta_values, mask_indices, box_values

def visualization(im, labels, bboxes, BATCH_CLASSES=None):
    text_bound = 3
    fontScale = 0.6
    thickness = 2

    BATCH_CLASSES = BATCH_CLASSES or [i for i in range(max(labels)+1)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx in xrange(len(bboxes)):
        cx, cy, w, h = [v for v in bboxes[idx]]
        cx1, cy1 = int(cx - w / 2.0), int(cy - h / 2.0)
        cx2, cy2 = int(cx + w / 2.0), int(cy + h / 2.0)
        cv2.rectangle(im, (cx1, cy1), (cx2, cy2), color=256, thickness=1)
        batch_id = labels[idx]
        label_name = BATCH_CLASSES[batch_id]
        txt = "Tag: {}".format(label_name)
        txtSize = cv2.getTextSize(txt, font, fontScale, thickness)[0]
        cv2.putText(im, txt,
                    (int((cx1+cx2-txtSize[0])/2.0),
                     cy1-text_bound\
                         if cy1-txtSize[1]-text_bound>text_bound else cy1+txtSize[1]+text_bound),
                    font, fontScale, 255, thickness=thickness)
    plt.imshow(im)



