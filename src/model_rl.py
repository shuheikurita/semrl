from collections import deque
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import logging

USE_Y_PRED=False

def softmax_mat(mat):
    r=[]
    for v in mat:
        e_x = np.exp(v - np.max(v))
        r.append(e_x / e_x.sum())
    return np.array(r)

logger = logging.getLogger(__name__)

def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

class RLAgent:

    def __init__(self, params, tasks, word_max, pred_max, init_wmat, init_pmat):
        # parameters

        self.replay_memory_size = 1000
        self.D = deque(maxlen=self.replay_memory_size)
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

        self.word_max = word_max        # with #ROOT
        self.word_max_input = word_max +1        # with #ROOT
        self.word_max_w_top = word_max +1        # with top-pred
        self.XPRED=1
        self.word_max_w_null = word_max + self.XPRED # at FNN, with #NULL and #NOT_PRED
        self.h_size=int(params["h_dim"][0])
        self.h_dep_size=int(params["h_dep_dim"][0])
        self.flag_size=int(params["flag_dim"][0])
        self.minibatch_size=1
        self.pred_num = pred_max
        self.pred_num_w_top = pred_max + 1
        self.init_wmat = init_wmat
        self.init_pmat = init_pmat
        self.tasks = tasks
        self.use_deparg = params["use_deparg"]
        self.embedding_size_dep = int(params["emb_dep_dim"][0])

    def init_model(self, params):
        # train operation
        #optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer(1e-4)

        # task-independent

        self.ids_w = tf.placeholder(tf.int64, [self.minibatch_size, self.word_max_input])
        self.ids_l = tf.placeholder(tf.int64, [self.minibatch_size, self.word_max_input])
        self.ids_p = tf.placeholder(tf.int64, [self.minibatch_size, self.word_max_input])

        if self.init_wmat is not None:
            self.embedding_size = self.init_wmat.shape[1]
            self.emb_w = tf.Variable(tf.constant(0.0, shape=self.init_wmat.shape), name="emb_w")
            self.emb_w_dep = tf.Variable(tf.constant(0.0, shape=(self.init_wmat.shape[0],self.embedding_size_dep) ), name="emb_w_dep")
            embedding_placeholder_w = tf.placeholder(tf.float32, self.init_wmat.shape)
            embedding_init_w = self.emb_w.assign(embedding_placeholder_w)
            embedding_init_w_dep = self.emb_w_dep.assign(embedding_placeholder_w[:,:self.embedding_size_dep])
        else:
            self.embedding_size = self.h_size
            self.emb_w = tf.get_variable("w_emb",
                [20000, self.embedding_size])
        if self.init_pmat is not None:
            assert self.embedding_size == self.init_pmat.shape[1]
            self.emb_p = tf.Variable(tf.constant(0.0, shape=self.init_pmat.shape), name="emb_p")
            self.emb_p_dep = tf.Variable(tf.constant(0.0, shape=(self.init_pmat.shape[0],self.embedding_size_dep) ), name="emb_p_dep")
            embedding_placeholder_p = tf.placeholder(tf.float32, self.init_pmat.shape)
            embedding_init_p = self.emb_p.assign(embedding_placeholder_p)
            embedding_init_p_dep = self.emb_p_dep.assign(embedding_placeholder_p[:,:self.embedding_size_dep])
        else:
            self.emb_p = tf.get_variable("p_emb",
                [50, self.embedding_size])

        # special arg. #ROOT and #NOT_PRED
        STDDEV=1./np.sqrt(100)
        h_pred_root = tf.Variable(tf.truncated_normal([1,self.h_size*2], stddev=STDDEV))

        self.use_lemma = params["use_lemma"]

        h_arg_null  = tf.Variable(tf.truncated_normal([1,self.h_size*2], stddev=STDDEV))
        h_arg_notp  = tf.Variable(tf.truncated_normal([1,self.h_size*2], stddev=STDDEV))

        h_w = tf.nn.embedding_lookup(self.emb_w, self.ids_w)
        h_l = tf.nn.embedding_lookup(self.emb_w, self.ids_l)
        h_p = tf.nn.embedding_lookup(self.emb_p, self.ids_p)
        if self.use_lemma:
            rnninput = tf.concat([h_w, h_l, h_p], 2)
        else:
            rnninput = tf.concat([h_w, h_p], 2)

        def stacked_lstm_no_sql():
            mlstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
            return rnn.dynamic_rnn(mlstm_cell, data, dtype=tf.float32)[0]

        def bi_lstm_no_sql(data):
            y1 = rnn.dynamic_rnn(lstm_cell, data,  dtype=tf.float32,scope='r1')[0]
            rdata = array_ops.reverse_sequence(input=data, seq_lengths=sql, seq_dim=1, batch_dim=0)
            y2 = rnn.dynamic_rnn(lstm_cell, rdata, dtype=tf.float32,scope='r2')[0]
            y2 = array_ops.reverse_sequence(input=y2, seq_lengths=sql, seq_dim=1, batch_dim=0)
            return [y1,y2]

        def simple_stack_bilstm(x,layers,num_hidden,timesteps,name):
            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
            x = tf.unstack(x, timesteps, 1)
            # Define lstm cells with tensorflow
            lstm_fw_cell = [rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=tf.AUTO_REUSE, name=name+"_f_"+str(i)) for i in range(layers)]
            lstm_bw_cell = [rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=tf.AUTO_REUSE, name=name+"_b_"+str(i)) for i in range(layers)]

            outputs, _, _ = rnn.stack_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
            return outputs

        lstm_layers=int(params["lstm_layers"][0])
        timesteps = self.word_max_input
        print("timesteps=",timesteps)
        if params["use_residual_connection"]:
            for ilay in range(lstm_layers):
                rnnoutput = simple_stack_bilstm(rnninput, 1, self.h_size, timesteps, "wlp"+str(ilay))
                rnnoutput = tf.stack(rnnoutput, axis=1)
                if ilay !=0:
                    rnninput=rnninput+rnnoutput
                else:
                    rnninput=rnnoutput
            self.batch_h_arg = rnninput
        else:
            rnninput = simple_stack_bilstm(rnninput, lstm_layers, self.h_size, timesteps, "wlp")
            self.batch_h_arg = tf.stack(rnninput, axis=1)

        h_arg = self.batch_h_arg[0]

        if params["prevent_update_lstm"]:
            h_arg = tf.stop_gradient(h_arg)

        if params["use_deparg"]:
            self.fnn_dim = self.h_size*4 + self.h_dep_size*2 + self.flag_size
        else:
            self.fnn_dim = self.h_size*4 + self.flag_size
        if params["use_deparg"] and params["bothinvrnndep"]:
            self.fnn_dim+=self.h_dep_size*2
        self.fnn_hidden_dim = int(params["fnn_hidden_dim"][0])

        XDIM = self.word_max
        YDIM = self.pred_num_w_top+self.XPRED
        self.XDIM=XDIM
        self.YDIM=YDIM

        setattr(self, "y_mask0", tf.placeholder(tf.float32 ,  [None, XDIM, YDIM],"y_mask0") )
        setattr(self, "y_mask1", tf.placeholder(tf.float32 ,  [None, XDIM, YDIM],"y_mask1") )
        setattr(self, "y_mask2", tf.placeholder(tf.float32 ,  [None, XDIM]               ,"y_mask2") )
        setattr(self, "y_mask3", tf.placeholder(tf.float32 ,  [None, self.word_max_input]               ,"y_mask3") )

        if True: # No - minibatch
            h_pred = h_arg[:-1]
            h_arg = h_pred
            h_arg = tf.concat([h_arg_null,h_arg],axis=0) # add special pred #NULL_arg and #NOT_PRED at the top of preds.
            h_arg = tf.concat([h_pred_root,h_arg],axis=0) # add special pred #ROOT at the top of preds.

            # [mini-batch, pred_num, arg_num, h_size]
            h_arg_ = tf.expand_dims(h_arg, 0)
            h_arg_ = tf.tile(h_arg_, tf.constant([XDIM,1,1]))
            h_pred_ = tf.expand_dims(h_pred, 1)
            h_pred_ = tf.tile(h_pred_, tf.constant([1,YDIM,1]))

            # task-dependent

            setattr(self, "current",   tf.placeholder(tf.int64, [None, XDIM, YDIM-self.XPRED],   "current")   )
            setattr(self, "ids_w_dep", tf.placeholder(tf.int64, [self.minibatch_size, XDIM, YDIM],"ids_w_dep") )
            setattr(self, "ids_l_dep", tf.placeholder(tf.int64, [self.minibatch_size, XDIM, YDIM],"ids_l_dep") )
            setattr(self, "ids_p_dep", tf.placeholder(tf.int64, [self.minibatch_size, XDIM, YDIM],"ids_p_dep") )
            flag_self_mention = tf.Variable(tf.truncated_normal([2,self.flag_size], stddev=STDDEV))

            if params["bothinvrnndep"]:
                h_deparg_null  = tf.Variable(tf.truncated_normal([1,1,self.h_dep_size*4], stddev=STDDEV))
                h_deparg_notp  = tf.Variable(tf.truncated_normal([1,1,self.h_dep_size*4], stddev=STDDEV))
            else:
                h_deparg_null  = tf.Variable(tf.truncated_normal([1,1,self.h_dep_size*2], stddev=STDDEV))
                h_deparg_notp  = tf.Variable(tf.truncated_normal([1,1,self.h_dep_size*2], stddev=STDDEV))
            h_current_null  = tf.Variable(tf.truncated_normal([1,1,self.flag_size], stddev=0.01))
            h_current_notp  = tf.Variable(tf.truncated_normal([1,1,self.flag_size], stddev=0.01))

            current   = getattr(self, "current")
            ids_w_dep = getattr(self, "ids_w_dep")
            ids_l_dep = getattr(self, "ids_l_dep")
            ids_p_dep = getattr(self, "ids_p_dep")

        for task in self.tasks:
            task="_"+task

            setattr(self, "y_only_one_deps"+task, tf.placeholder(tf.int64,
                    [None, XDIM, YDIM], "y_only_one_deps"+task) ) # positions of predicates
            setattr(self, "y"+task, tf.placeholder(tf.int64,
                    [None, XDIM, YDIM], "y"+task) ) # positions of predicates
            setattr(self, "rewards"+task, tf.placeholder(tf.float32,
                    [None, XDIM], "rewards"+task) ) # positions of predicates

            y_only_one_deps = getattr(self, "y_only_one_deps"+task)
            y = getattr(self, "y"+task)
            rewards = getattr(self, "rewards"+task)


            h_w_dep = tf.nn.embedding_lookup(self.emb_w_dep, ids_w_dep)
            #h_l_dep = tf.nn.embedding_lookup(self.emb_w_dep, ids_l_dep)
            h_p_dep = tf.nn.embedding_lookup(self.emb_p_dep, ids_p_dep)
            #rnninput_dep = tf.concat([h_w_dep[0], h_l_dep[0], h_p_dep[0]], 2)
            rnninput_dep = tf.concat([h_w_dep[0], h_p_dep[0]], 2)
            lstm_layers_dep=int(params["lstm_layers_dep"][0])

            if params["invrnndep"]:
                rnninput_dep = tf.transpose(rnninput_dep, perm=[1,0,2])
                timesteps = XDIM
            else:
                timesteps = YDIM
            print("rnndep timesteps=",timesteps)

            if params["multitask_sepdep"]:
                rnnoutput_dep = simple_stack_bilstm(rnninput_dep,lstm_layers_dep, self.h_dep_size, timesteps, "dep"+task)
                logger.info("You are using sepdep!")
            else:
                rnnoutput_dep = simple_stack_bilstm(rnninput_dep,lstm_layers_dep, self.h_dep_size, timesteps, "dep")
            if params["invrnndep"]:
                h_deparg = tf.stack(rnnoutput_dep, axis=0)
            else:
                h_deparg = tf.stack(rnnoutput_dep, axis=1)

            assert not (params["invrnndep"] and params["bothinvrnndep"])
            if params["bothinvrnndep"]:
                rnninput_dep = tf.transpose(rnninput_dep, perm=[1,0,2])
                timesteps = XDIM
                rnnoutput_dep = simple_stack_bilstm(rnninput_dep,lstm_layers_dep, self.h_dep_size, timesteps, "invdep")
                h_deparg_ = tf.stack(rnnoutput_dep, axis=0)
                h_deparg = tf.concat([h_deparg,h_deparg_], axis=2)

            # remove the last arg that is always #NULL or #EOS and meaningless for arg-pred analysis.
            h_deparg = h_deparg[:,:-1,:]

            if params["prevent_update_lstmdep"]:
                h_deparg = tf.stop_gradient(h_deparg)

            # add #NULL and #NON_PRED
            h_deparg_1 = tf.tile(h_deparg_null, tf.constant([XDIM,1,1]))
            h_deparg_2 = tf.tile(h_deparg_notp, tf.constant([XDIM,1,1]))
            if self.XPRED==2:
                raise NotImplemented
                h_deparg = tf.concat([h_deparg_1,h_deparg_2,h_deparg], axis=1)
            else:
                h_deparg = tf.concat([h_deparg_1,h_deparg], axis=1)

            # current
            h_current=tf.nn.embedding_lookup(flag_self_mention, current)
            h_current = h_current[0]
            # current doen't remove the last arg that is always #NULL or #EOS and meaningless for arg-pred analysis.
            # add #NULL and #NON_PRED
            h_current_1 = tf.tile(h_current_null, tf.constant([XDIM,1,1]))
            h_current_2 = tf.tile(h_current_notp, tf.constant([XDIM,1,1]))
            if self.XPRED==2:
                h_current = tf.concat([h_current_1,h_current_2,h_current], axis=1)
            else:
                h_current = tf.concat([h_current_1,h_current], axis=1)

            #fnninput = tf.concat([h_arg_, h_pred_, h_current], 2)
            if params["use_deparg"]:
                fnninput = tf.concat([h_arg_, h_pred_, h_current, h_deparg], 2)
            else:
                fnninput = tf.concat([h_arg_, h_pred_, h_current], 2)
            fnninput = tf.reshape(fnninput, [XDIM*YDIM, self.fnn_dim])

            # FNN

            setattr(self, "W_fc1"+task, tf.Variable(tf.truncated_normal([self.fnn_dim, self.fnn_hidden_dim], stddev=0.01)) )
            setattr(self, "b_fc1"+task, tf.Variable(tf.zeros([self.fnn_hidden_dim])) )
            setattr(self, "W_out"+task, tf.Variable(tf.truncated_normal([self.fnn_hidden_dim,1], stddev=0.01)) )
            #b_out = tf.Variable(tf.zeros([1]))

            W_fc1 = getattr(self, "W_fc1"+task)
            b_fc1 = getattr(self, "b_fc1"+task)
            W_out = getattr(self, "W_out"+task)
            y_mask0 = getattr(self, "y_mask0")
            y_mask1 = getattr(self, "y_mask1")
            y_mask2 = getattr(self, "y_mask2")

            temp = tf.matmul(fnninput,W_fc1)
            h_fc1 = tf.nn.relu(temp + b_fc1)
            if params["droppout_rate_fnn"][0]!=0.0:
                h_fc1 = tf.nn.dropout(h_fc1,keep_prob=(1.-float(params["droppout_rate_fnn"][0])))
            else:
                logger.info("NO dropuout in FNN")

            if params["use_fnn3"]:
                setattr(self, "W_fc2"+task, tf.Variable(tf.truncated_normal([self.fnn_hidden_dim, self.fnn_hidden_dim], stddev=0.01)) )
                setattr(self, "b_fc2"+task, tf.Variable(tf.zeros([self.fnn_hidden_dim])) )
                W_fc2 = getattr(self, "W_fc2"+task)
                b_fc2 = getattr(self, "b_fc2"+task)
                temp = tf.matmul(h_fc1,W_fc2)
                h_fc1 = tf.nn.relu(temp + b_fc2)
                if params["droppout_rate_fnn"][0]!=0.0:
                    h_fc1 = tf.nn.dropout(h_fc1,keep_prob=(1.-float(params["droppout_rate_fnn"][0])))

            y = tf.matmul(h_fc1, W_out)# + b_out

            if params["use_highway_fnn"]:
                setattr(self, "W_out2"+task, tf.Variable(tf.truncated_normal([self.fnn_dim,1], stddev=0.01)) )
                W_out2 = getattr(self, "W_out2"+task)
                temp = tf.matmul(fnninput,W_out2)
                y=y+temp

            if params["use_highway_fnn2"]:
                setattr(self, "hw2_W_fc1"+task, tf.Variable(tf.truncated_normal([self.fnn_dim, self.fnn_hidden_dim], stddev=0.01)) )
                setattr(self, "hw2_b_fc1"+task, tf.Variable(tf.zeros([self.fnn_hidden_dim])) )
                setattr(self, "hw2_W_out"+task, tf.Variable(tf.truncated_normal([self.fnn_hidden_dim,1], stddev=0.01)) )
                hw2_W_fc1 = getattr(self, "hw2_W_fc1"+task)
                hw2_b_fc1 = getattr(self, "hw2_b_fc1"+task)
                hw2_W_out = getattr(self, "hw2_W_out"+task)
                temp = tf.matmul(fnninput,hw2_W_fc1)
                hw2_h_fc1 = tf.nn.relu(temp + hw2_b_fc1)
                if params["droppout_rate_fnn"][0]!=0.0:
                    hw2_h_fc1 = tf.nn.dropout(hw2_h_fc1,keep_prob=(1.-float(params["droppout_rate_fnn"][0])))
                temp = tf.matmul(hw2_h_fc1,hw2_W_out)
                y=y+temp

            raw_scores = tf.reshape(y, [XDIM,YDIM])
            raw_scores = raw_scores * y_mask0[0]
            raw_scores_cc = raw_scores + y_mask1[0]
            scores_sq       = tf.sigmoid(raw_scores_cc)
            scores_cc       = tf.nn.softmax(raw_scores_cc)

            each_loss_cc = tf.nn.softmax_cross_entropy_with_logits_v2(\
                    logits=raw_scores_cc,labels=y_only_one_deps)
            each_loss_cc = each_loss_cc * y_mask2[0]
            loss_cc      = tf.reduce_mean(each_loss_cc)

            y = tf.reshape(y, [XDIM,YDIM])
            #each_loss_sq = tf.reduce_sum(tf.square(raw_scores-y),axis=1)
            each_loss_sq = tf.reduce_sum(tf.square(scores_sq-y),axis=1)
            each_loss_sq = each_loss_sq * y_mask2[0]
            loss_sq      = tf.reduce_mean(each_loss_sq)

            loss_pg = tf.reduce_mean(each_loss_cc * rewards)  # reward guided loss

            setattr(self, "each_loss_cc" +task, each_loss_cc )
            setattr(self, "each_loss_sq" +task, each_loss_sq )
            setattr(self, "loss_cc"      +task, loss_cc )
            setattr(self, "loss_sq"      +task, loss_sq )
            setattr(self, "loss_pg"      +task, loss_pg )
            setattr(self, "scores_cc"    +task, scores_cc )
            setattr(self, "scores_sq"    +task, scores_sq )
            setattr(self, "raw_scores"+task, raw_scores )
            setattr(self, "raw_scores_cc"+task, raw_scores_cc )

            setattr(self, "train_op_cc"+task, optimizer.minimize(loss_cc))
            setattr(self, "train_op_pg"+task, optimizer.minimize(loss_pg))
            #setattr(self, "train_op_sq"+task, optimizer.minimize(loss_sq))

        # saver
        self.saver = tf.train.Saver(max_to_keep=0)

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.init_wmat is not None:
            self.sess.run([embedding_init_w,embedding_init_w_dep], feed_dict={embedding_placeholder_w: self.init_wmat})
        if self.init_pmat is not None:
            self.sess.run([embedding_init_p,embedding_init_p_dep], feed_dict={embedding_placeholder_p: self.init_pmat})

    def supervised(self, nninput_supervised, square=False):
        if USE_Y_PRED:
            task, ids_wp, ids_wp_dep, preds_at, y, y_mask, current, y_preds = nninput_supervised
        else:
            task, ids_wp, ids_wp_dep, preds_at, y, y_mask, current = nninput_supervised
        task="_"+task
        feed_dict={\
                getattr(self,"ids_w")               : [ids_wp[0]],
                getattr(self,"ids_p")               : [ids_wp[2]],
                getattr(self,"ids_w_dep")      : [ids_wp_dep[0]],
                getattr(self,"ids_p_dep")      : [ids_wp_dep[2]],
                getattr(self,"y_mask0")        : [y_mask[0]],
                getattr(self,"y_mask1")        : [y_mask[1]],
                getattr(self,"y_mask2")        : [y_mask[2]],
                getattr(self,"current")        : [current],
                }
        if self.use_deparg:
            feed_dict[getattr(self,"ids_w_dep")] = [ids_wp_dep[0]]
            feed_dict[getattr(self,"ids_p_dep")] = [ids_wp_dep[2]]
        if self.use_lemma:
            feed_dict[getattr(self,"ids_l")] = [ids_wp[1]]
            if self.use_deparg:
                feed_dict[getattr(self,"ids_l_dep")] = [ids_wp_dep[1]]
        try:
            x=0
            if square:
                feed_dict[getattr(self,"y"+task)] = [y]
                x = self.sess.run([\
                    getattr(self,"loss_sq"+task),
                    getattr(self,"raw_scores"+task),
                    getattr(self,"each_loss_sq"+task),
                    getattr(self,"train_op_sq"+task)],
                    feed_dict)
            else:
                feed_dict[getattr(self,"y_only_one_deps"+task)] = [y]
                x = self.sess.run([\
                    getattr(self,"loss_cc"+task),
                    getattr(self,"raw_scores_cc"+task),
                    getattr(self,"each_loss_cc"+task),
                    getattr(self,"train_op_cc"+task)],
                    feed_dict)
        except:
            import ipdb;ipdb.set_trace()
            raise
        return x

    def pglearn(self, nninput, actions_mat, rewards, square=False):
        task, ids_wp, ids_wp_dep, preds_at, y_mask, current = nninput
        task="_"+task
        feed_dict={\
                getattr(self,"ids_w")               : [ids_wp[0]],
                getattr(self,"ids_l")               : [ids_wp[1]],
                getattr(self,"ids_p")               : [ids_wp[2]],
                getattr(self,"ids_w_dep")      : [ids_wp_dep[0]],
                getattr(self,"ids_l_dep")      : [ids_wp_dep[1]],
                getattr(self,"ids_p_dep")      : [ids_wp_dep[2]],
                getattr(self,"y_mask0")        : [y_mask[0]],
                getattr(self,"y_mask1")        : [y_mask[1]],
                getattr(self,"y_mask2")        : [y_mask[2]],
                getattr(self,"current")        : [current],
                getattr(self,"rewards"+task)        : [rewards],
                }
        try:
            x=0
            if square:
                raise NotImplemented
                feed_dict[getattr(self,"y"+task)] = [y]
                x = self.sess.run([\
                    getattr(self,"loss_sq"+task),
                    getattr(self,"raw_scores"+task),
                    getattr(self,"each_loss_sq"+task),
                    getattr(self,"train_op_sq"+task)],
                    feed_dict)
            else:
                feed_dict[getattr(self,"y_only_one_deps"+task)] = [actions_mat]
                x = self.sess.run([\
                    getattr(self,"loss_pg"+task),
                    getattr(self,"raw_scores_cc"+task),
                    getattr(self,"each_loss_cc"+task),
                    getattr(self,"train_op_pg"+task)],
                    feed_dict)
        except:
            import ipdb;ipdb.set_trace()
            raise
        return x

    def Q_values_random(self, nninput): # for debug
        task, ids_wp, ids_wp_dep, preds_at, y_mask, current = nninput
        raw_scores = np.random.random([self.pred_num_w_top, self.word_max_w_null])
        raw_scores = raw_scores * y_mask[0]
        return softmax_mat(raw_scores)

    def Q_values(self, nninput):
        task, ids_wp, ids_wp_dep, preds_at, y_mask, current = nninput
        task="_"+task
        feed_dict={\
                getattr(self,"ids_w")               : [ids_wp[0]],
                getattr(self,"ids_p")               : [ids_wp[2]],
                getattr(self,"y_mask0")        : [y_mask[0]],
                getattr(self,"y_mask1")        : [y_mask[1]],
                getattr(self,"y_mask2")        : [y_mask[2]],
                getattr(self,"current")        : [current],
                }
        if self.use_deparg:
            feed_dict[getattr(self,"ids_w_dep")] = [ids_wp_dep[0]]
            feed_dict[getattr(self,"ids_p_dep")] = [ids_wp_dep[2]]
        if self.use_lemma:
            feed_dict[getattr(self,"ids_l")] = [ids_wp[1]]
            if self.use_deparg:
                feed_dict[getattr(self,"ids_l_dep")] = [ids_wp_dep[1]]
        # Q(state, action) of all actions

        return self.sess.run([\
            getattr(self,"scores_cc"+task)],
            feed_dict)[0]

    def Q_values_pred(self, nninput):
        task, ids_wp, ids_wp_dep, preds_at, y_mask, current = nninput
        task="_"+task
        feed_dict={\
                getattr(self,"ids_w")               : [ids_wp[0]],
                getattr(self,"ids_l")               : [ids_wp[1]],
                getattr(self,"ids_p")               : [ids_wp[2]],
                }

        return self.sess.run([\
            getattr(self,"yp"+task)],
            feed_dict)[0]

    def store_experience(self, nninput, actions, rewards, state_1, terminal):
        self.D.append((nninput, actions, rewards, state_1, terminal))

    def experience_replay(self):
        np.random.shuffle(self.D)
        for nninput, actions, rewards, state_1, terminal in self.D:
            self.pglearn(nninput, actions, rewards, False)
        self.D=[]

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self, model_path, epoch=None):
        if epoch:
            self.saver.save(self.sess, model_path, global_step=epoch)
        else:
            self.saver.save(self.sess, model_path)
