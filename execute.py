# coding=utf-8
import time
import logging
import tensorflow as tf

from collections import defaultdict

from cnn import CNN
from data_helper import load_data, load_embedding, batch_iter, create_valid, build_vocab
from utils import convert_map_to_array, average_gradients, tower_loss, cal_predictions

#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "data/train.txt", "train corpus file")
tf.flags.DEFINE_string("test_file", "data/test.txt", "test corpus file")
tf.flags.DEFINE_string("word_file", "data/words.txt", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "data/vectors.txt", "vector file")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5", "filter size of cnn")
tf.flags.DEFINE_integer("embedding_size", 150, "embedding size")
tf.flags.DEFINE_integer("sequence_len", 80, "embedding size")
tf.flags.DEFINE_integer("num_filters", 128, "the number of filter in every layer")
tf.flags.DEFINE_float("dropout", 0.5, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 256, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 100, "epoches")
tf.flags.DEFINE_integer("evaluate_every", 2000, "run evaluation")
tf.flags.DEFINE_integer("l2_reg_lambda", 0.0, "l2 regulation")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.9, "use memory rate")

FLAGS = tf.flags.FLAGS
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger ---------------------------------------
logging.basicConfig(format="%(message)s", level=logging.INFO)
#----------------------------- define a logger end -----------------------------------

#------------------------------------load data -------------------------------
word2idx, idx2word = build_vocab(FLAGS.word_file)
label2sents = load_data(FLAGS.train_file, word2idx, FLAGS.sequence_len)
logging.info("load train data finish")
total_x, total_y = convert_map_to_array(label2sents)
logging.info("convert finish")
train_x, valid_x = create_valid(total_x)
train_y, valid_y = create_valid(total_y)
num_classes = len(label2sents.keys())
embedding = load_embedding(FLAGS.embedding_size, filename=FLAGS.embedding_file)
test_label2sents = load_data(FLAGS.test_file, word2idx, FLAGS.sequence_len)
logging.info("load test data finish")
test_x, test_y = convert_map_to_array(test_label2sents, is_shuffle=False)
logging.info("convert finish")
#----------------------------------- load data end ----------------------

#----------------------------------- cal filter_size --------------------------------------
filter_sizes = [int(filter_size.strip()) for filter_size in FLAGS.filter_sizes.strip().split(",")]
#----------------------------------- cal filter_size end ----------------------------------

#----------------------------------- step -------------------------------------------------
def run_step(sess, cnn, batch_x, batch_y, dropout=1., is_optimizer=True):
    start_time = time.time()
    feed_dict = {
        input_x:batch_x,
        input_y:batch_y, 
        cnn.dropout:dropout
    }

    if is_optimizer:
        step, cur_logits, cur_loss, cur_acc, _ = sess.run([global_step, logits, loss, acc, train_op], feed_dict)
    else:
        step, cur_logits, cur_loss, cur_acc = sess.run([global_step, logits, loss, acc], feed_dict)

    elapsed_time = time.time() - start_time
    return step, cur_logits, cur_loss, cur_acc, elapsed_time

#----------------------------------- step end ---------------------------------------------

#----------------------------------- validate model --------------------------------------
def validate_model(sess, cnn, valid_x, valid_y):
    start_time = time.time()
    batches = batch_iter(zip(valid_x, valid_y), FLAGS.batch_size, 1)
    total_loss, total_acc, total_elapsed_time = 0, 0, 0
    idx = 0
    for batch in batches:
        batch_x, batch_y = zip(*batch)
        step, cur_logits, cur_loss, cur_acc, elapsed_time = run_step(sess, cnn, batch_x, batch_y, is_optimizer=False)
        total_loss += cur_loss
        total_acc += cur_acc
        total_elapsed_time += elapsed_time
        idx += 1

    aver_loss = 1. * total_loss / idx
    aver_acc = 1. * total_acc / idx
    aver_elapsed_time = 1. * total_elapsed_time / idx
    validate_elapsed_time = time.time() - start_time
    logging.info("validation loss:%s, acc:%s, %6.7f secs/batch_size, total elapsed time: %6.7f"%(aver_loss, aver_acc, aver_elapsed_time, validate_elapsed_time))
    
#----------------------------------- validate model end ----------------------------------

#----------------------------------- execute train ---------------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            cnn = CNN(FLAGS.batch_size, FLAGS.sequence_len, embedding, FLAGS.embedding_size, filter_sizes, FLAGS.num_filters, num_classes, l2_reg_lambda=FLAGS.l2_reg_lambda)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)

            input_x = tf.placeholder(tf.int32, [None, FLAGS.sequence_len])
            input_y = tf.placeholder(tf.int32, [None,])
            logits, loss, acc = tower_loss(cnn, input_x, input_y)

            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            sess.run(tf.initialize_all_variables())

            # load batch data
            batches = batch_iter(zip(train_x, train_y), FLAGS.batch_size, FLAGS.epoches)
            for batch in batches:
                batch_x, batch_y = zip(*batch)
                step, cur_logits, cur_loss, cur_acc, elapsed_time = run_step(sess, cnn, batch_x, batch_y, FLAGS.dropout)
                logging.info("%s steps, loss:%s, acc:%s, %6.7f secs/batch_size"%(step, cur_loss, cur_acc, elapsed_time))
                cur_step = tf.train.global_step(sess, global_step)

                if cur_step % FLAGS.evaluate_every == 0:
                    logging.info("************** start to evaluate model *****************")
                    validate_model(sess, cnn, valid_x, valid_y)

            # test model
            logging.info("********************* start to test model ****************************")
            validate_model(sess, cnn, test_x, test_y)
#----------------------------------- execute train end -----------------------------------
