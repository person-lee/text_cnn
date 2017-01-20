#coding=utf-8

import numpy as np
import tensorflow as tf

def convert_map_to_array(label2sents, is_shuffle=True):
    """
    convert map to arry
    """
    input_x, input_y = [], []
    label_idx = 0
    for each_items in label2sents.items():
        input_x.extend(each_items[1])
        input_y.extend(np.ones(len(each_items[1]), dtype=int) * label_idx)
        label_idx += 1

    if is_shuffle:
        np.random.seed(12345)
        shuffle_idx = np.random.permutation(np.arange(len(input_y)))
        input_x, input_y = [input_x[idx] for idx in shuffle_idx], [input_y[idx] for idx in shuffle_idx]

    return input_x, input_y


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def tower_loss(cnn, input_x, input_y):
    logits = cnn.inference(input_x)
    total_loss = cnn.loss(logits, input_y)
    accuracy = cnn.accuracy(logits, input_y)
    return logits, total_loss, accuracy

def cal_predictions(cnn, input_x):
    logits = cnn.inference(input_x)
    predictions = tf.argmax(logits, 1, name="predictions")
    return predictions
