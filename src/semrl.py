#!/usr/bin/env python
# -*- coding: utf-8 -*-
# a main file for SDP

import argparse
import logging
import pickle
from collections import OrderedDict

from loaddata import *
import trainmodel

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s/%(name)s[%(levelname)s]: %(message)s',
                                level=logging.DEBUG)

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument('--train', help='Train the parser using a train and test (val) file (CONLL format)', action="store_true")
    parser.add_argument('--test', help='Train the parser using a train and test (val) file (CONLL format)', action="store_true")
    parser.add_argument('--analysis', help='Analysis', action="store_true")

    parser.add_argument('--parse', help='Parse some text of CONLL format', action="store_true")

    parser.add_argument('--parsing',      help='Specify the mode, Parsing SDP graph', action="store_true")
    parser.add_argument('--labeling',     help='Specify the mode, Label inference', action="store_true")
    parser.add_argument('--attach_label', help='Specify the mode, Attach SDP labels on the parsed result', action="store_true")

    parser.add_argument('--input',        help='Input file(s) for train/parse text file (CONLL format). (This shuould be used with --datanpy.)', nargs="+")
    parser.add_argument('--output_json', help='Output file(s) for train/parse text file (CONLL format).', nargs="+")

    parser.add_argument('--datapkl', help='dataset pkl file name', nargs=1, default=[None])
    parser.add_argument('--datanpy', help='npy embedding file name', nargs=1, default=[None])

    parser.add_argument('--tasks',     help='Task name(s) of dm/pas/psd', nargs='+', default=[None])
    parser.add_argument('--inittasks', help='Task name(s) for initialising the neural model. This option is for the multi-task models. dm/pas/psd', nargs='+', default=[None])
    parser.add_argument('--splits',    help='Split name(s) of datasets like train/dev/test/ood', nargs='+', default=[None])

    parser.add_argument('--lstm_layers', help='The number of layers of Bi-LSTM', nargs=1, default=[3])
    parser.add_argument('--lstm_layers_dep', help='The number of layers of Bi-LSTM', nargs=1, default=[1])
    parser.add_argument('--lr', help='learning rate', nargs=1, default=[0.01])
    parser.add_argument('--epoch_max', help='Number of epochs for training', nargs=1, default=[10])
    parser.add_argument('--epoch_from', help='Starting epoch number', nargs=1, default=[0])
    parser.add_argument('--droppout_rate_fnn', help='', nargs=1, default=[0.0])
    parser.add_argument('--droppout_rate_rnn', help='', nargs=1, default=[0.0])
    parser.add_argument('--fnn_hidden_dim', help='', nargs=1, default=[2048])
    parser.add_argument('--h_dim', help='', nargs=1, default=[128])
    parser.add_argument('--h_dep_dim', help='', nargs=1, default=[128])
    parser.add_argument('--emb_dep_dim', help='', nargs=1, default=[64])
    parser.add_argument('--flag_dim', help='', nargs=1, default=[64])

    parser.add_argument('--multitask_sepdep', help='separate dep lstm', action="store_true")

    parser.add_argument('--rl_training', help='DQN training', action="store_true")
    parser.add_argument('--rl_tran_max', help='Number of trans for pg', nargs=1, default=[4])
    parser.add_argument('--loss_square', help='loss type', action="store_true")
    parser.add_argument('--gold_easy_fast', help='Gold oracle', nargs=1, default=[0])

    parser.add_argument('--savepath',  help='Path to save models', nargs=1, default=[None])
    parser.add_argument('--saveresult',  help='Path to save results', nargs=1, default=[None])
    parser.add_argument('--loadresult',  help='Path to load results', nargs=1, default=[None])
    parser.add_argument('--loadresult_transpose',  help='Transpose results when loaded', action="store_true")
    parser.add_argument('--savelabels',  help='Path to save labels', nargs=1, default=[None])
    parser.add_argument('--loadlabel_transpose',  help='Transpose labels when loaded', action="store_true")
    parser.add_argument('--loadlabel_dm',  help='Path to load labels', nargs=1, default=[None])
    parser.add_argument('--loadlabel_pas',  help='Path to load labels', nargs=1, default=[None])
    parser.add_argument('--loadlabel_psd',  help='Path to load labels', nargs=1, default=[None])
    parser.add_argument('--loadpath',  help='Path of load models', nargs=1, default=[None])
    parser.add_argument('--loadepoch', help='Load epoch number', nargs=1, default=[0])

    parser.add_argument('--debug_no_init_model', help='Skip init model', action="store_true")
    parser.add_argument('--debug_no_train', help='Skip training', action="store_true")
    parser.add_argument('--debug_one_sentence', help='Debug with only one sentence', action="store_true")
    parser.add_argument('--debug_onstate_learning', help='Debug with only one sentence', action="store_true")
    parser.add_argument('--debug_dqn_pseudo_Q', help='', action="store_true")
    parser.add_argument('--debug_mb_limit', help='', nargs=1, default=[-1])
    parser.add_argument('--skip_validation', help='Skip validation per epoch while training', action="store_true")

    parser.add_argument('--invrnndep', help='invrnndep', action="store_true")
    parser.add_argument('--bothinvrnndep', help='invrnndep', action="store_true")
    parser.add_argument('--donotusenopred', help='0,1,2', nargs=1, default=[0])
    parser.add_argument('--use_deparg', help='', action="store_true")
    parser.add_argument('--use_lemma', help='', action="store_true")
    parser.add_argument('--pg_non_pred_at', help='No pg for learning of non-predicate words.', action="store_true")
    parser.add_argument('--pg_notfinish', help='0 or -1', nargs=1, default=[0])
    parser.add_argument('--prevent_update_lstm', help='', action="store_true")
    parser.add_argument('--prevent_update_lstmdep', help='', action="store_true")

    parser.add_argument('--use_residual_connection', help='', action="store_true")
    parser.add_argument('--use_highway_fnn', help='', action="store_true")
    parser.add_argument('--use_highway_fnn2', help='', action="store_true")
    parser.add_argument('--use_fnn3', help='', action="store_true")

    parser.add_argument('--mix_lemmma_and_words', help='Mixed dict of word and lemma', action="store_true")

    opts = parser.parse_args()
    params = OrderedDict(vars(opts))

    #tasks=["dm","pas","psd"]
    tasks=params["tasks"]
    inittasks = tasks if params["inittasks"][0] is None else params["inittasks"]
    loadepoch = int(params["loadepoch"][0])

    if params["labeling"] or params["attach_label"]:
        params["rl_training"]=False

    if params["datapkl"] != [None]: # load prepared data
        path = params["datapkl"][0]
        logger.info("Load from "+path)
        data = pickle.load(open(path,"rb"))
    elif params["datanpy"] != [None]:
        assert len(set(params["output_json"]))==len(set(params["input"]))
        data = Datasets(None)
        data.tasks=params["tasks"][:]
        data.splits=params["splits"][:] if len(params["splits"])==len(params["input"]) and params["splits"] != [None] \
            else [str(i) for i in range(len(params["input"]))]
        params["splits"]=data.splits[:]
        data=load_emb(data, params["datanpy"][0])
        params["iomap"]={}
        def fileindex(task,split,splits):
            return len(splits)*data.tasks.index(task)+splits.index(split)
        for task in data.tasks:
            params["iomap"][task]={}
            for split in data.splits:
                file = params["input"][fileindex(task,split,data.splits)]
                outf = params["output_json"][fileindex(task,split,data.splits)]
                logger.info('Add conll file: '+file)
                data.read(file,task,split,mix_lemmma_and_words=params["mix_lemmma_and_words"])
                params["iomap"][task][split]={"in":file,"out":outf}
    else:
        print("--datapkl or --datanpy required!")
        quit()
    if params["parse"]:
        data.init_nn_inputs_limited(False,True)
    else:
        data.init_nn_inputs(False,True)

    splits=params["splits"]

    if params["attach_label"]:
        assert params["loadresult"][0]
        logger.info("*** LABELING TEST MODE *** ")
        logger.info("Epoch %d"%loadepoch)
        trainmodel.valid(params,tasks,splits,data,loadepoch)
        quit()

    import model_rl as nnmodel
    import model_labels as nnlabel

    logger.info('\t\t\t Initializing the model ... ')
    if params["parsing"]:
        nnm = nnmodel.RLAgent(params, inittasks,data.max_words, data.max_preds, data.wmat, data.pmat)
        if not params["debug_no_init_model"]:
            nnm.init_model(params)
        params["model"]=nnm

    if params["labeling"]:
        nnm = nnlabel.RLAgent(params, inittasks,data.max_words, data.max_preds, data.wmat, data.pmat, data.labl_sets)
        if not params["debug_no_init_model"]:
            nnm.init_model(params)
        params["model_labeling"]=nnm

    if params["loadpath"][0]:
        loadpath=(params["loadpath"][0]+"_%03d.ckpt")%loadepoch
        logger.info("### LOADING MODEL FROM:"+loadpath+"Epoch :"+str(loadepoch))
        nnm.load_model(loadpath)

    if opts.train:
        logger.info("*** TRAINING MODE *** ")
        trainmodel.train(params,tasks,splits,data)
    if opts.test:
        logger.info("*** TESTING MODE *** ")
        if params["analysis"]:
            trainmodel.analysis(params,tasks,splits,data,loadepoch)
        else:
            trainmodel.valid(params,tasks,splits,data,loadepoch)

    logger.info("End.")
