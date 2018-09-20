from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import hgtk
import pickle
import sys
import ph_model
import ph_data_reader
import numpy as np
import os
from numpy import dot
from numpy.linalg import norm

flags = tf.app.flags
flags.DEFINE_string("input_vec", './data/NN-numalp-space-ep200em200min24.vec', "input vector location.")
flags.DEFINE_string("model_save_path", './saved_model/', "Directory to write the model.")
flags.DEFINE_string("cuda_device", '2', "Cuda device to run")
flags.DEFINE_integer("embedding_size", 200, "Size of Input vector embedding size")
flags.DEFINE_float("learning_rate", 0.1, "Initial learning rate.")
flags.DEFINE_float("dropout", 0.8, "drop out")
flags.DEFINE_float("max_grad_norm", 5.0, "max gradient norm")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_integer("highway_layer", 7, "num of highway layer")
flags.DEFINE_integer("epochs_to_train", 12, "epochs to train")
flags.DEFINE_boolean("inference_mode", False, "Inference mode for OOV")
flags.DEFINE_integer("lr_decay_step", 10000, "every decay step lr decay")
FLAGS = flags.FLAGS

os.environ['LANG'] = 'ko_KR.utf8'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device

# Vector preprocessing.
ph_data_reader.vector_preprocessing(FLAGS.input_vec)
#train data manipulation.
exlist =[ '자본주의', '백과사전','순간이동', '오른손', '파란색', '달라진다', '일어나기','경제학'
         ,'쓰러졌다', '인민공화국', '북유럽', '아이언맨', '명예훼손']

# Vectors are already L2 normalized, so not necessary to norm.
def cosine_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

# in: word exclude list, ret : dict values of excluded word.
# ex_vec_list : == list shape(n, array(1,200) ) x[n][0] = word, x[n][1] = (1,200)
def word_exclude_for_test(ex_wd_list):
    ex_wd_dct={}
    for i in ex_wd_list:
        ex_wd_dct.update(ph_data_reader.exclude_word(i))
    return ex_wd_dct

def inference_word(session, valid_model, word):
    evx =ph_data_reader.unk_padder(word)
    # Evaluation. fv shape = (1,word_emb_size)
    fv2 = session.run(
        valid_model.fv2
    , {
        valid_model.input  : evx
    })
    fv2=fv2[0]
    return fv2

# Function for evaluation, similarity.
# Usage : evx_str= OOV word for inference, evy_str= test word for similarity.
def word_distance(session, valid_model, evx_str, evy_str):
    #evx (1,29) evy (1,word_emb_size)
    #evx : ph_model input character padded.
    #evy : known word, from w2v voc vector.
    #fv2 : ph_model output of evx char padding.
    evy =ph_data_reader.w2v(evy_str)
    evy=evy[0]
    fv2 = inference_word(session, valid_model, evx_str)
    sim = cosine_sim(fv2,evy)
    print(evy_str +' sim=%6.5f' %(sim), end=' ')

# Usage : w2v_known_test('일어','나기')
def w2v_known_test(a,b, c=0.5,d=0.5 ):
    evy =ph_data_reader.w2v(a)
    evy2 =ph_data_reader.w2v(b)
    ev3 = evy*c + evy2*d
    # ev3 for composed vector rate of c,d. compare to a+b known word vector.
    x = cosine_sim(ev3[0], ph_data_reader.w2v(a+b)[0])
    print(a+'/'+b+' w2v composition sim: ' +str(x))

def validate_model():
    # validate generated OOV vector with known words.
    ex1 = '수박아이스크림'
    word_distance(session, valid_model, ex1, '수박')
    word_distance(session, valid_model, ex1, '아이스크림')
    word_distance(session, valid_model, ex1, '아이')
    word_distance(session, valid_model, ex1, '스크림')
    word_distance(session, valid_model, ex1, '아이스')
    word_distance(session, valid_model, ex1, '크림\n')

    # validate character level generated vector by word similarity.
    word_distance(session, valid_model, '전기자동차', '전기')
    word_distance(session, valid_model, '전기자동차', '자동차')
    word_distance(session, valid_model, '오토바이헬멧', '오토바이')
    word_distance(session, valid_model, '오토바이헬멧', '헬멧\n')
    word_distance(session, valid_model, '아이스크림', '아이스크림')
    word_distance(session, valid_model, '인공지능', '인공지능\n')

def main(_):
    global exlist
    saver=None
    
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
        param_init=0.05
        initializer = tf.random_uniform_initializer(-param_init, param_init)
        # Vector preprocessing.
        ph_data_reader.vector_preprocessing(FLAGS.input_vec)
        ph_data_reader.processing(btch_size=FLAGS.batch_size)
        with tf.variable_scope("Model", initializer=initializer):

            train_model = ph_model.inference_graph2(batch_size=FLAGS.batch_size
                                                    , word_embed_size=FLAGS.embedding_size
                                                    , max_word_length=ph_data_reader.mwlen
                                                    , num_highway_layers=FLAGS.highway_layer)
            train_model.update(ph_model.loss_graph2(batch_size=FLAGS.batch_size
                                                    ,fv2=train_model.fv2, word_embed_size=FLAGS.embedding_size))
            train_model.update(ph_model.training_graph2(loss=train_model.loss, learning_rate=FLAGS.learning_rate 
                                                        ,lr_decay_step=FLAGS.lr_decay_step
                                                        ,max_grad_norm=FLAGS.max_grad_norm))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver()

        with tf.variable_scope("Model", reuse=True):
            # Batch size is 1 - one inference at one time. (self limiatation)
            valid_model = ph_model.inference_graph2(batch_size=1,word_embed_size=FLAGS.embedding_size
                                                    ,max_word_length=ph_data_reader.mwlen
                                                    ,num_highway_layers=FLAGS.highway_layer)
            valid_model.update(ph_model.loss_graph2(valid_model.fv2, batch_size=1, word_embed_size=FLAGS.embedding_size))

        
        if not FLAGS.inference_mode:
            
            #train data manipulation.
            tf.global_variables_initializer().run()
            session.run(train_model.clear_char_embedding_padding)

            # Excluded for training exlist words for OOV sim test.
            ex_word_dict = word_exclude_for_test(exlist)
            print('deleted word : ', end='')
            for i in ex_word_dict:
                print(ex_word_dict[i][0].decode('utf-8') ,end=' ')
            print()

            # xs shape (2587, batch_size, max_word_len(mwlen)), ys shape (2587, batch_size, word_emb_size)
            xs, ys = ph_data_reader.processing(btch_size=FLAGS.batch_size)
            
            #after batch data made, add excluded words to wdict for w2v test.
            ph_data_reader.update_wdict(ex_word_dict)

            # Count for log print.
            count=0
            count2=0
            for i in range(FLAGS.epochs_to_train):
                print('-epoch %d ' %(i))
                for xline, yline in zip(xs,ys):
                    #xline (20,29) yline (20,word_emb_size)
                    count+=1
                    #prob for drop out in training.
                    loss, step, _,_, learning_rate = session.run([
                        train_model.loss, 
                        train_model.global_step ,  
                        train_model.clear_char_embedding_padding,
                        train_model.train_op,
                        train_model.learning_rate2
                    ], {
                        train_model.input : xline,
                        train_model.ansy : yline,
                        train_model.prob : FLAGS.dropout
                    })
                    if count%3000==0:
                        print('step=%6d, loss=%6.8f, lr=%6.8f' %(step, loss, learning_rate))

                count2+=1
                if count2%1==0:
                    validate_oov()

                if count2%1==0:
                    print('--test results for OOV & Known word Similarity---')
                    for ii in exlist:
                        word_distance(session, valid_model, ii, ii)
                    print('--------------------------------------------')
        
            saver.save(session, os.path.join(FLAGS.model_save_path, "model.ckpt"), global_step=step)
            print('model saved, training done')

        else : 
            checkpoint = tf.train.latest_checkpoint(FLAGS.model_save_path)
            saver.restore(session, checkpoint)
#            print('input words')
#            sys.stdout.flush()
            try:
                for line in sys.stdin:
                    line=line.replace('\n','')
                    vec = inference_word(session, valid_model, line)
                    sys.stdout.write(line +' ')
                    for i in vec:
                        sys.stdout.write(str(i) + ' ')
                    sys.stdout.write('\n')
                    
            except KeyboardInterrupt:
                sys.stdout.flush()
                pass


if __name__ == "__main__":
    tf.app.run()

