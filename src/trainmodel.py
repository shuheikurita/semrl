
import numpy as np
import logging
import json
import pickle

from gen_minibatches import *
import parsemodel
from util_std import *

global ps
global params

best_fval   =0.0
best_vacj   =0.0
best_vacs   =0.0
best_vacs0  =0.0
best_vact   =0.0
best_vact2  =0.0

USE_Y_PRED=False

logger = logging.getLogger(__name__)


def getpredfromsent(sent):
    sentid,words,lemmas,poss,tops,preds,pa=sent
    return preds

def show_scores(golds,preds):
    assert type(golds)==list
    assert type(preds)==list
    assert type(golds[0])==np.ndarray
    assert type(preds[0])==np.ndarray
    print(preds[1].shape)
    print(golds[1].shape)
    if len(preds)!=len(golds):
        logger.info("preds and golds num mismatch! : %d vs %d"%(len(preds),len(golds)))
    assert preds[0].shape==golds[0].shape

    logger.info("### Score ALL:")
    f,p,r = parsemodel.prf_numpy2(golds,preds)
    logger.info("F/P/R: %f / %f / %f"%(f*100.,p*100.,r*100.))

    logger.info("### Score MACRO-averaged:")
    f,p,r = parsemodel.prf_numpy2_macro(golds,preds)
    logger.info("F/P/R: %f / %f / %f"%(f*100.,p*100.,r*100.))


def show_scores_label(golds,preds):
    assert type(golds)==list
    assert type(preds)==list
    assert type(golds[0])==np.ndarray
    assert type(preds[0])==np.ndarray
    print(preds[0].shape,golds[0].shape)
    assert preds[0].shape==golds[0].shape
    if len(preds)!=len(golds):
        logger.info("preds and golds num mismatch! : %d vs %d"%(len(preds),len(golds)))

    logger.info("### Score ALL:")
    acc = parsemodel.prf_numpy_class(golds,preds)
    logger.info("Accuracy: %f "%(acc*100.))

def get_f_array(golds,preds):
    return np.array([parsemodel.prf_numpy_LAS_micro(gmat,pmat)[0] for gmat,pmat in zip(golds,preds)])

def show_scores_LAS(golds,preds,masks=None):
    assert type(golds)==list
    assert type(preds)==list
    assert type(golds[0])==np.ndarray
    assert type(preds[0])==np.ndarray
    assert preds[0].shape==golds[0].shape
    if len(preds)!=len(golds):
        logger.info("preds and golds num mismatch! : %d vs %d"%(len(preds),len(golds)))

    logger.info("### Score ALL:")
    f,p,r = parsemodel.prf_numpy_LAS(golds,preds,masks)
    logger.info("F/P/R: %f / %f / %f"%(f*100.,p*100.,r*100.))

    logger.info("### Score MACRO-averaged:")
    f,p,r = parsemodel.prf_numpy_LAS_macro(golds,preds,masks)
    logger.info("F/P/R: %f / %f / %f"%(f*100.,p*100.,r*100.))

def train(lparams, tasks, splits, data):
    global params
    params=lparams

    # TRAIN MODEL
    logger.info('Begin training of the model...')

    n_epochs=int(params["epoch_max"][0])
    epoch   =int(params["epoch_from"][0])
    if params["labeling"]:
        nnm=params["model_labeling"]
    else:
        nnm=params["model"]
    square = params["loss_square"]
    NOTFINISH = int(params["pg_notfinish"][0])
    DEBUG_MB_LIMIT = int(params["debug_mb_limit"][0])

    # CPU memory saving
    current_shape = (data.max_words,data.max_preds_nn)
    deparg_shape  = (data.max_words,data.max_preds_nn+1)
    y_shape       = (data.max_words,data.max_preds_nn+1)
    current_to_predargs = parsemodel.current_to_predargs

    tasks_states=[]
    while epoch < n_epochs:
        epoch = epoch + 1
        logger.info("\n\t\t\t Epoch %d" %(epoch))
        if params["rl_training"]:
            tasks_states=[]
            for task in tasks:
                tasks_states.append(rl_training(params, data, task, splits[0]))
            states = []
            for ts in tasks_states:
                states += ts
            if not params["debug_no_train"]:
                np.random.shuffle(states)
                rl_training_exec(params, data, states, NOTFINISH)
        else:
            if (not params["labeling"]) or len(tasks_states)==0:
                tasks_states=[]
                for task in tasks:
                    if not params["debug_no_train"]:
                        if params["labeling"]:
                            states_ = suprvised_training_label(params, data, task, splits[0])
                        else:
                            states_ = suprvised_training(params, data, task, splits[0])
                        tasks_states.append(states_)
                states = []
                for ts in tasks_states:
                    states += ts
            if not params["debug_no_train"]:
                countstate=0
                np.random.shuffle(states)
                for state in states:
                    try:
                        if params["labeling"]:
                            name,nninput_supervised,counttrans = state
                            task, [ids_w, ids_l, ids_p], labl_t3, y_mask = nninput_supervised
                            labl_t3 = label_to_t3(*labl_t3)
                            nninput_supervised = task, [ids_w, ids_l, ids_p], labl_t3, y_mask
                            loss=nnm.supervised(nninput_supervised,square)
                        else:
                            # CPU memory saving
                            name,nninput_supervised,counttrans = state
                            task,[ids_w, ids_l, ids_p], [_,_,_], preds_at, y, mask_, current = nninput_supervised
                            ids_w_dep = current_to_predargs(current, ids_w, deparg_shape, data.widv, addeos=True)
                            ids_l_dep = current_to_predargs(current, ids_l, deparg_shape, data.widv, addeos=True)
                            ids_p_dep = current_to_predargs(current, ids_p, deparg_shape, data.pidv, addeos=True)
                            current_pad = pad_current(current,current_shape)
                            nninput_supervised = task,[ids_w, ids_l, ids_p], [ids_w_dep, ids_l_dep, ids_p_dep], preds_at, y, mask_, current_pad
                            loss=nnm.supervised(nninput_supervised,square)
                    except:
                        import ipdb; ipdb.set_trace()
                        raise
                    logger.info("[train][%d]%6.2f%%, sent: #%d, loss: %f"%(countstate,countstate*100./len(states),name[0],loss[0]))
                    if loss[0]>1000:
                        import ipdb; ipdb.set_trace()
                    countstate+=1
                    if countstate==DEBUG_MB_LIMIT: break


        savepath = "%s_%03d.ckpt"%(params["savepath"][0],epoch)
        logger.info("### SAVEING MODEL TO: %s"%savepath)
        nnm.save_model(savepath)

        if not params["skip_validation"]:
            for task in tasks:
                if params["pred_detect"]:
                    pred_at_predicted = validation_preds_at(params, data, task, splits[1])
                elif params["labeling"]:
                    for valsplit in splits[1:]:
                        logger.info("\n\t\t\t Validation (ONCE), epoch: %d, task: %s, split: %s" %(epoch,task,valsplit))
                        predicts,allpredicts = validation_label(params, data, task, valsplit)
                        show_scores_label(data.labls[task][valsplit],predicts)
                        if params["savelabels"][0]:
                            with open(params["savelabels"][0]+"."+task+"."+valsplit+".ep"+str(epoch)+".pkl", mode='wb') as f:
                                pickle.dump(allpredicts, f)
                else:
                    logger.info("\n\t\t\t Validation (ONCE), epoch: %d, task: %s" %(epoch,task))
                    results = validation_once(params, data, task, splits[1])
                    predicts = [ pred[0] for pred in results ]
                    valsplit=splits[-1]
                    logger.info("\n\t\t\t Validation (ONCE), epoch: %d, task: %s, split: %s" %(epoch,task,valsplit))
                    show_scores(data.golds[task][valsplit],predicts)
                    logger.info("\n\t\t\t Validation (ONCE) END, epoch: %d, task: %s, split: %s" %(epoch,task,valsplit))
                    if params["saveresult"][0]:
                        #with open(params["saveresult"][0]+"."+task+"."+valsplit+".pkl", mode='wb') as f:
                        with open(params["saveresult"][0]+".epoch"+str(epoch)+"."+task+"."+valsplit+".pkl", mode='wb') as f:
                            pickle.dump(predicts, f)

        logger.info("")

    print("End.")

def Q_values_to_sampled_action(Q_value, mask, currentshape, max_actions, preds_at, pg_non_pred_at): # without pred_dones
    actions=[]
    #import ipdb; ipdb.set_trace()
    XPRED=1
    for i,v in enumerate(Q_value*mask):
        if i<currentshape[0]:
            if i in preds_at:
                action = np.random.choice(np.arange(max_actions), p=v)
                assert action < currentshape[1]+XPRED
            else:
                if pg_non_pred_at:
                    action=1
                else:
                    action = np.random.choice(np.arange(max_actions), p=v)
                    assert action < currentshape[1]+XPRED
        else:
            action=0
        actions.append(action)
    return np.array(actions)

def rl_training(params, data, task, split):

    ## init datasets
    currents=[]
    flags_finish=[]
    mask_preds=[]
    flags_pred_dones=[]
    for z in data.inits[task][split]:
        currents.append([z.copy()])
        flags_finish.append([0])
    for z in data.masks[task][split]:
        mask_preds.append(np.copy(z[2]))
        flags_pred_dones.append([False for x in range(data.max_preds_nn)])

    ## training
    init_states=[]
    for ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,pred_dones in zip(\
            data.inputs_wid[task][split],
            data.inputs_lid[task][split],
            data.inputs_pid[task][split],
            data.preds_at[task][split],
            data.golds[task][split],
            data.masks[task][split],
            data.datasets[task][split],
            currents,
            mask_preds,
            flags_pred_dones\
            ):
        pred_dones1=pred_dones[:]
        state_t = [ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current[0],mask_pred,pred_dones,pred_dones1]
        init_states.append((state_t,task))

    return init_states

def rl_training_exec(params, data, init_states, NOTFINISH):

    current_shape = (data.max_words,data.max_preds_nn)
    deparg_shape  = (data.max_words,data.max_preds_nn+1)
    current_to_predargs = parsemodel.current_to_predargs

    countsent=0
    allsent=len(init_states)

    model = params["model"]
    XDIM=model.XDIM
    YDIM=model.YDIM
    pg_non_pred_at = params["pg_non_pred_at"]
    debug_dqn_pseudo_Q = params["debug_dqn_pseudo_Q"]
    dqn_tran_max = int(params["dqn_tran_max"][0])
    stat_tran_num=[]
    assert params["use_deparg"]
    for state_t_1,task in init_states:

        #while not terminal:
        for tran_num in range(dqn_tran_max):
            state_t = state_t_1

            # execute action in environment
            nninput = state_to_nninput_deparg(state_t, data, task, current_shape, deparg_shape, current_to_predargs)

            if not debug_dqn_pseudo_Q:
                try:
                    Q_values=model.Q_values(nninput)
                except:
                    import ipdb; ipdb.set_trace()
                    raise
            else:
                Q_values=model.Q_values_random(nninput)

            ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,_,_,_ = state_t
            action_t = Q_values_to_sampled_action(Q_values, mask[0], current.shape, YDIM, getpredfromsent(sent), pg_non_pred_at)

            # observe environment
            state_t_1, reward_t, terminal = update_state_w_reward(state_t,action_t,data, pg_non_pred_at, XDIM, NOTFINISH)

            actions_mat = np.zeros((XDIM,YDIM))
            actions_mat[np.arange(XDIM),action_t]=1

            # store experience
            model.store_experience(nninput, actions_mat, reward_t, None, terminal)

            if terminal: break
        stat_tran_num.append(tran_num)

        logger.info("End - Get parsing, tran_num: %d, sent: #%d"%(tran_num,sent[0]))
        # experience replay
        model.experience_replay()

        logger.info("%6.2f%% sent: #%d"%(countsent*100./allsent,sent[0]))

        countsent+=1
        if params["debug_one_sentence"]: break
    logger.info("End - PG training, mean_tran_num: %f"%(np.mean(tran_num)))
    return


def valid3(lparams, tasks, splits, data, loadresults, epochs):
    global params
    params=lparams

    goldlabelss=[]
    goldlabels_=[]
    predictss=[]
    predicts_=[]

    for task,loadresult,epoch in zip(tasks,loadresults,epochs):
        for valsplit in splits:
            #import ipdb; ipdb.set_trace()
            logger.info("\n\t\t\t Validation with %s, %s" %(task,valsplit))
            with open(loadresult+".epoch"+str(epoch)+"."+task+"."+valsplit+".pkl", mode='rb') as f:
                where_to_label = pickle.load(f)
                logger.info("%d sentences result loaded." %(len(where_to_label)))
                assert len(data.golds[task][valsplit])==len(where_to_label)
                if params["loadresult_transpose"]:
                    where_to_label = [x.transpose() for x in where_to_label]
            show_scores(data.golds[task][valsplit],where_to_label)
            if params["loadlabel_"+task][0]:
                #with open(params["loadlabel_"+task][0]+"."+task+"."+valsplit+".ep"+params["loadepoch"][0]+".pkl", mode='rb') as f:
                with open(params["loadlabel_"+task][0], mode='rb') as f:
                    labels = pickle.load(f)
                    logger.info("%d sentences labels loaded." %(len(labels)))
                    assert len(data.golds[task][valsplit])==len(labels)
                show_scores_label(data.labls[task][valsplit],labels)
                predicts = validation_attach_label(data.labls[task][valsplit],data.golds[task][valsplit],where_to_label,labels)
                show_scores_LAS(data.labls[task][valsplit],predicts)
                predictss+=predicts
                predicts_.append(predicts)
                goldlabelss+=data.labls[task][valsplit]
                goldlabels_.append(data.labls[task][valsplit])
    show_scores_LAS(goldlabelss,predictss)
    predicts_.append(predictss)
    goldlabels_.append(goldlabelss)
    return goldlabels_,predicts_

def valid(lparams, tasks, splits, data, epoch=0):
    global params
    params=lparams

    for task in tasks:
        for valsplit in splits:
            logger.info("\n\t\t\t Validation with %s, %s" %(task,valsplit))
            if params["labeling"] or params["attach_label"]:
                if params["loadresult"][0]:
                    with open(params["loadresult"][0]+".epoch"+str(epoch)+"."+task+"."+valsplit+".pkl", mode='rb') as f:
                        where_to_label = pickle.load(f)
                        logger.info("%d sentences result loaded." %(len(where_to_label)))
                        assert len(data.golds[task][valsplit])==len(where_to_label)
                        if params["loadresult_transpose"]:
                            where_to_label = [x.transpose() for x in where_to_label]
                    show_scores(data.golds[task][valsplit],where_to_label)
                    if params["loadlabel_"+task][0]:
                        with open(params["loadlabel_"+task][0], mode='rb') as f:
                            labels = pickle.load(f)
                            logger.info("%d sentences labels loaded." %(len(labels)))
                            assert len(data.golds[task][valsplit])==len(labels)
                            if params["loadlabel_transpose"]:
                                labels = [x.transpose() for x in labels]
                        show_scores_label(data.labls[task][valsplit],labels)
                        predicts = validation_attach_label(data.labls[task][valsplit],data.golds[task][valsplit],where_to_label,labels)
                        show_scores_LAS(data.labls[task][valsplit],predicts)
                    else:
                        predicts,allpredicts = validation_label(params, data, task, valsplit, where_to_label)
                        show_scores_LAS(data.labls[task][valsplit],predicts)
                else:
                    print("*** loadlabel_"+task+" NOT FOUND!")
                    predicts,allpredicts = validation_label(params, data, task, valsplit)
                    show_scores_label(data.labls[task][valsplit],predicts)
                    if params["savelabels"][0]:
                        with open(params["savelabels"][0]+"."+task+"."+valsplit+".ep"+str(epoch)+".pkl", mode='wb') as f:
                            pickle.dump(allpredicts, f)

            else:
                results = validation_once(params, data, task, valsplit)
                predicts = [ pred[0].astype('int64')  for pred in results ]
                if data.golds is not None:
                    show_scores(data.golds[task][valsplit],predicts)
                logger.info("\n\t\t\t Parse END with %s, %s" %(task,valsplit))
                if params["saveresult"][0]:
                    print_results(params["saveresult"][0]+".epoch"+str(epoch)+"."+task+"."+valsplit+".parsed.txt",
                                  data,data.datasets[task][valsplit],data.golds[task][valsplit],predicts)
                    with open(params["saveresult"][0]+".epoch"+str(epoch)+"."+task+"."+valsplit+".pkl", mode='wb') as f:
                        pickle.dump(predicts, f)
                pickle.dump([predicts,params["iomap"][task][valsplit],data], open("temp.pkl","wb"))
                if params["output_json"][0]:
                    logger.info("Write out to %s"%params["iomap"][task][valsplit])
                    writeout(predicts,params["iomap"][task][valsplit],data)

def analysis(lparams, tasks, splits, data, epoch=0):
    global params
    params=lparams

    for task in tasks:
        for valsplit in splits:
            logger.info("\n\t\t\t Parse with %s, %s" %(task,valsplit))
            if params["labeling"] or params["attach_label"]:
                if params["loadresult"][0]:
                    with open(params["loadresult"][0]+".epoch"+str(epoch)+"."+task+"."+valsplit+".pkl", mode='rb') as f:
                        where_to_label = pickle.load(f)
                        logger.info("%d sentences result loaded." %(len(where_to_label)))
                        assert len(data.golds[task][valsplit])==len(where_to_label)
                    show_scores(data.golds[task][valsplit],where_to_label)
                    if params["loadlabel_"+task][0]:
                        with open(params["loadlabel_"+task][0], mode='rb') as f:
                            labels = pickle.load(f)
                            logger.info("%d sentences labels loaded." %(len(labels)))
                            assert len(data.golds[task][valsplit])==len(labels)
                        show_scores_label(data.labls[task][valsplit],labels)
                        predicts = validation_attach_label(data.labls[task][valsplit],data.golds[task][valsplit],where_to_label,labels)
                        show_scores_LAS(data.labls[task][valsplit],predicts)
                    else:
                        predicts,allpredicts = validation_label(params, data, task, valsplit, where_to_label)
                        show_scores_LAS(data.labls[task][valsplit],predicts)
                else:
                    print("*** loadlabel_"+task+" NOT FOUND!")
                    predicts,allpredicts = validation_label(params, data, task, valsplit)
                    show_scores_label(data.labls[task][valsplit],predicts)
                    if params["savelabels"][0]:
                        with open(params["savelabels"][0]+"."+task+"."+valsplit+".ep"+params["loadepoch"][0]+".pkl", mode='wb') as f:
                            pickle.dump(allpredicts, f)

            else:
                results = validation_analysis(params, data, task, valsplit)
                predicts = [ pred[0].astype('int64') for pred in results ]
                if data.golds is not None:
                    show_scores(data.golds[task][valsplit],predicts)
                logger.info("\n\t\t\t Parse END with %s, %s" %(task,valsplit))
                if params["saveresult"][0]:
                    print_results(params["saveresult"][0]+".epoch"+str(epoch)+"."+task+"."+valsplit+".parsed.txt",
                                  data,data.datasets[task][valsplit],data.golds[task][valsplit],predicts)
                    with open(params["saveresult"][0]+".epoch"+str(epoch)+"."+task+"."+valsplit+".pkl", mode='wb') as f:
                        pickle.dump(predicts, f)
                pickle.dump([predicts,params["iomap"][task][valsplit],data], open("temp.pkl","wb"))
                if params["output_json"][0]:
                    logger.info("Write out to %s"%params["iomap"][task][valsplit])
                    writeout(predicts,params["iomap"][task][valsplit],data)

def writeout(predicts,iomap,data):
    predicts=[p.tolist() for p in predicts]
    json.dump(predicts,open(iomap["out"],"w"))

def pickleit(filename,obj,condition=True):
    if condition:
        with open(filename, mode='wb') as f:
            pickle.dump(obj, f)

def clear_results(filename):
    with open(filename, mode='wt') as f:
        pass

def print_results(filename,data,sents,golds,preds):
    with open(filename, mode='wt') as f:
        for sent,gold,pred in zip(sents,golds,preds):
            data.fshow(f,sent)
            f.write("***gold\n")
            f.write(str(gold))
            f.write("***pred\n")
            f.write(str(pred))
            f.write("\n\n")

def update_state(state,action_t):
    ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,preds_dones = state
    for i,act in  enumerate(action_t):
        current[i,act]=1
    countnull=0
    for i,a in enumerate(action_t): # without pred_dones
        assert i<current.shape[0]  # pred num limit
        if a==0:
            countnull+=1          # NULL transition
            continue
        assert a<current.shape[1]            # arg (words) num limit
        current[i,a-1]=1
    terminal = 1 if countnull==current.shape[0] else 0

# pg core
def update_state_w_reward(state,action_t,data,pg_non_pred_at,XDIM,NOTFINISH=0):
    ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,pred_dones,pred_dones1 = state
    reward=[]
    res=gold-current
    next_current=np.copy(current)
    next_mask_pred=np.copy(mask_pred)
    next_pred_dones=pred_dones[:]
    next_pred_dones1=pred_dones1[:]
    XPRED=1
    #import ipdb;ipdb.set_trace()
    for i,(g,c,r,n,m,d,d1,a) in enumerate(zip(gold,current,res,next_current,next_mask_pred,next_pred_dones,next_pred_dones1,action_t)):
        assert i<current.shape[0]  # pred num limit
        if a==0:
            if d1:
                next_pred_dones1[i]=False
                next_pred_dones[i]=True
#                    next_mask_pred[i]=0.
                reward.append(0)
            elif d:
                reward.append(0)    # already finished
            else:
                if np.min(c-g)==0:
                    next_pred_dones1[i]=True
                    reward.append(1)    # finish now
                    next_pred_dones[i]=True # giving up
                else:
                    reward.append(NOTFINISH)    # not finished
        else:
            assert a-XPRED<current.shape[1]            # arg (words) num limit
            next_current[i,a-XPRED]=1
            if r[a-XPRED]==1:
                reward.append(1)
            elif g[a-XPRED]==1:
                reward.append(0)
            else:
                reward.append(-1)

    for _ in range(XDIM-gold.shape[0]):
        reward.append(0)
    reward=np.array(reward,dtype=np.float32)
    terminal = 1 if counttrue(next_pred_dones)>=current.shape[0] else 0

    state = ids_w,ids_l,ids_p,preds_at,gold,mask,sent,next_current,next_mask_pred,next_pred_dones,next_pred_dones1

    return state, reward, terminal

def pad_current(current,current_shape):
    current_pad = np.zeros(current_shape)
    current_pad[0:current.shape[0],0:current.shape[1]] += current
    return current_pad

def state_to_nninput_supervised(state_t, data, task, y_shape, current_shape, deparg_shape, square=False, donotusenopred=False, gold_easy_fast=0):
    ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,preds_dones = state_t
    if square:
        y,next_current,next_mask_pred,next_preds_dones=\
                parsemodel.sample_gold_square(gold,current[0],data.max_words_nn,data.max_preds_nn,mask_pred[0],preds_dones[0])
    else:
        y,next_current,next_mask_pred,next_preds_dones=\
                parsemodel.sample_gold_once(gold,current[0], y_shape, mask_pred[0],preds_dones[0],preds_at,donotusenopred,gold_easy_fast)
    current_real=current[0]
    current[0]=next_current
    mask_pred[0]=next_mask_pred
    preds_dones[0]=next_preds_dones
    mask_=mask[0:2]+[next_mask_pred]+[mask[3]]
    terminal = 1 if y is None else 0
    sentid,words,lemmas,poss,tops,preds,pa=sent
    if USE_Y_PRED:
        y_preds = np.array([ [0.,1.] if i in preds else [1.,0.] for i in range(data.max_words_input)], dtype=np.float32)

        nninput_supervised = task,[ids_w, ids_l, ids_p], [None,None,None], preds_at, y, mask_, current_real, y_preds
    else:
        nninput_supervised = task,[ids_w, ids_l, ids_p], [None,None,None], preds_at, y, mask_, current_real
    state_t_1 = ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,preds_dones

    return state_t_1, nninput_supervised, terminal

def state_to_nninput(state_t, data, task, current_shape, deparg_shape):
    ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,_,_,_ = state_t
    current_pad = np.zeros(current_shape)
    current_pad[0:current.shape[0],0:current.shape[1]] += current
    nninput = [task,[ids_w,ids_l,ids_p], [None,None,None], preds_at, mask, current_pad]
    return nninput

def state_to_nninput_deparg(state_t, data, task, current_shape, deparg_shape, current_to_predargs):
    ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,_,_,_ = state_t
    ids_w_dep = current_to_predargs(current, ids_w, deparg_shape, data.widv, addeos=True)
    ids_l_dep = current_to_predargs(current, ids_l, deparg_shape, data.widv, addeos=True)
    ids_p_dep = current_to_predargs(current, ids_p, deparg_shape, data.pidv, addeos=True)
    current_pad = np.zeros(current_shape)
    current_pad[0:current.shape[0],0:current.shape[1]] += current
    nninput = [task,[ids_w,ids_l,ids_p], [ids_w_dep,ids_l_dep,ids_p_dep], preds_at, mask, current_pad]
    return nninput

def nninput_to_singleminibatch(nninput):
    [task,[ids_w,ids_l,ids_p], [ids_w_dep,ids_l_dep,ids_p_dep], preds_at, mask, current_pad] = nninput
    nninput = [task,[[ids_w],[ids_l],[ids_p]], [[ids_w_dep],[ids_l_dep],[ids_p_dep]], preds_at, mask, current_pad]
    return nninput

def suprvised_training(params, data, task, split):
    square = params["loss_square"]
    donotusenopred = int(params["donotusenopred"][0])
    gold_easy_fast = int(params["gold_easy_fast"][0])

    ## init datasets
    currents=[]
    mask_preds=[]
    flags_pred_dones=[]
    for z in data.inits[task][split]:
        currents.append([z.copy()])
    for z in data.masks[task][split]:
        mask_preds.append([np.copy(z[2])])
        flags_pred_dones.append([[False for x in range(data.max_preds_nn)]])

    ## training
    allsent=len(data.golds[task][split])
    states_t=[]
    for ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,pred_dones in zip(\
            data.inputs_wid[task][split],
            data.inputs_lid[task][split],
            data.inputs_pid[task][split],
            data.preds_at[task][split],
            data.golds[task][split],
            data.masks[task][split],
            data.datasets[task][split],
            currents,
            mask_preds,
            flags_pred_dones\
            ):
        state = ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,pred_dones
        states_t.append(state)

    current_shape = (data.max_words,data.max_preds_nn)
    deparg_shape  = (data.max_words,data.max_preds_nn+1)
    y_shape       = (data.max_words,data.max_preds_nn+1)
    current_to_predargs = parsemodel.current_to_predargs

    list_nninput_supervised=[]
    for counttrans in range(data.max_trans):
        countsent=0
        if len(states_t)==0: break
        states_t_1 = []
        for state_t in states_t:
            state_t_1, nninput_supervised, terminal = state_to_nninput_supervised(state_t, data, task, y_shape, current_shape, deparg_shape, square, donotusenopred, gold_easy_fast)
            if not terminal:
                states_t_1.append(state_t_1)
            else:
                continue

            if params["debug_onstate_learning"]:
                try:
                    task,[ids_w, ids_l, ids_p], [_,_,_], preds_at, y, mask_, current = nninput_supervised
                    ids_w_dep = current_to_predargs(current, ids_w, deparg_shape, data.widv, addeos=True)
                    ids_l_dep = current_to_predargs(current, ids_l, deparg_shape, data.widv, addeos=True)
                    ids_p_dep = current_to_predargs(current, ids_p, deparg_shape, data.pidv, addeos=True)
                    current_pad = pad_current(current,current_shape)
                    nninput_supervised = task,[ids_w, ids_l, ids_p], [ids_w_dep, ids_l_dep, ids_p_dep], preds_at, y, mask_, current_pad
                    loss=params["model"].supervised(nninput_supervised,square)
                except:
                    import ipdb; ipdb.set_trace()
                    raise
                ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,pred_dones = state_t
                logger.info("[%d]%6.2f%% sent: #%d loss: %f"%(counttrans,countsent*100./allsent,sent[0],loss[0]))
                if loss[0]>100:
                    import ipdb; ipdb.set_trace()
            else:
                list_nninput_supervised.append(([sent[0]],nninput_supervised,counttrans))
                logger.info("[%d]%6.2f%% sent: #%d"%(counttrans,countsent*100./allsent,sent[0]))

            countsent+=1
            if params["debug_one_sentence"]: break
        states_t = states_t_1
    return list_nninput_supervised

def label_to_t3(labl,pad_shape,label_num):
    t3 = np.zeros((pad_shape[0],pad_shape[1],label_num))
    for i,lx in enumerate(labl):
        for j,v in enumerate(lx):
            #t3[i,np.arange(labl.shape[1]),lx]=1
            t3[i,j,v]=1 if v !=-1 else 0
    return t3


def suprvised_training_label(params, data, task, split):
    donotusenopred = int(params["donotusenopred"][0])

    ## init datasets
    currents=[]
    mask_preds=[]
    flags_pred_dones=[]
    for z in data.inits[task][split]:
        currents.append([z.copy()])
    for z in data.masks[task][split]:
        mask_preds.append([np.copy(z[2])])
        flags_pred_dones.append([[False for x in range(data.max_preds_nn)]])

    ## training
    allsent=len(data.golds[task][split])
    list_nninput_supervised=[]
    countsent=0
    counttrans=0
    nnl=params["model_labeling"]
    gold_shape=(nnl.XDIM,nnl.YDIM)
    for ids_w,ids_l,ids_p,labl,gold,mask,sent in zip(\
            data.inputs_wid[task][split],
            data.inputs_lid[task][split],
            data.inputs_pid[task][split],
            data.labls[task][split],
            data.golds[task][split],
            data.masks[task][split],
            data.datasets[task][split],
            ):
        #state = ids_w,ids_l,ids_p,preds_at,gold,mask,sent,current,mask_pred,pred_dones
        #labl_t3 = label_to_t3(labl,gold_shape,len(data.labl_sets[task]))
        labl_t3 = (labl,gold_shape,len(data.labl_sets[task]))
        gold_pad = np.zeros(gold_shape)
        try:
            gold_pad[0:gold.shape[0],0:gold.shape[1]] += gold
        except:
            print(gold.shape)
            print(gold_shape)
            raise
        y_mask = [gold_pad]
        nninput_supervised = task, [ids_w, ids_l, ids_p], labl_t3, y_mask

        if params["debug_onstate_learning"]:
            try:
                if params["labeling"]:
                    task, [ids_w, ids_l, ids_p], labl_t3, y_mask = nninput_supervised
                    labl_t3 = label_to_t3(*labl_t3)
                    nninput_supervised = task, [ids_w, ids_l, ids_p], labl_t3, y_mask
                    loss=nnl.supervised(nninput_supervised)
                else:
                    loss=params["model_labeling"].supervised(nninput_supervised)
            except:
                import ipdb; ipdb.set_trace()
                raise
            #import ipdb; ipdb.set_trace()
            logger.info("[label]%6.2f%% sent: #%d loss: %f"%(countsent*100./allsent,sent[0],loss[0]))
            if loss[0]>100:
                import ipdb; ipdb.set_trace()
        else:
            list_nninput_supervised.append(([sent[0]],nninput_supervised,counttrans))
            logger.info("[label]%6.2f%% sent: #%d"%(countsent*100./allsent,sent[0]))

        countsent+=1
        if params["debug_one_sentence"]: break
    return list_nninput_supervised

def validation_preds_at(params, data, task, split):

    if params["debug_dqn_pseudo_Q"]:
        logger.info("Using gold preds_at")
        return data.preds_at[task][split]

    ## init datasets
    model = params["model"]

    ## validation
    preds_ats=[]
    countsent=0
    allsent=len(data.golds[task][split])
    for iw,il,ip,init,mask,sent in zip(\
            data.inputs_wid[task][split],
            data.inputs_lid[task][split],
            data.inputs_pid[task][split],
            data.inits[task][split],
            data.masks[task][split],
            data.datasets[task][split]):
        ids_w=iw
        ids_l=il
        ids_p=ip
        preds_at=None
        sentid,words,lemmas,poss,tops,preds,pa=sent

        nninput = [task,[ids_w,ids_l,ids_p], [None,None,None], preds_at, mask, None]
        try:
            scores=model.Q_values_pred(nninput)
        except:
            import ipdb; ipdb.set_trace()
            raise
        preds_at=[]
        for i,s in enumerate(scores):
            if not i<len(words): break  # pred num limit
            if s[1]>s[0]:
                preds_at.append(i)
        preds_ats.append(preds_at)
        logger.info("[valid][pred]%6.2f%% sent: #%d"%(countsent*100./allsent,sent[0]))
        countsent+=1
        #if countsent==10: break
        if params["debug_one_sentence"]: break


    sumpred,sumgold,sumandv=0,0,0
    for sent,pl in zip(data.datasets[task][split],preds_ats):
        sentid,words,lemmas,poss,tops,preds,pa=sent
        print("[preds_at] gold:",preds)
        print("[preds_at] pred:",pl)
        print("")
        for g in preds:
            if g in pl:
                sumandv+=1
            sumgold+=len(preds)
        for p in pl:
            sumpred+=len(pl)
    precision=divwoe(sumandv,sumpred)
    recall   =divwoe(sumandv,sumgold)
    f,p,r = divwoe(2.*(precision*recall),(precision+recall)),precision,recall
    logger.info("[pred_at] F/P/R: %f / %f / %f"%(f*100.,p*100.,r*100.))
    logger.info("[pred_at] TP,sumpred,sumgold,all(ref.) = %d,%d,%d"%(sumandv,sumpred,sumgold))

    return preds_ats

def validation_once(params, data, task, split):

    ## init datasets
    currents=[]
    flags_finish=[]
    flags_pred_dones=[]
    for z in data.inits[task][split]:
        currents.append([z.copy()])
        flags_finish.append([0])
        flags_pred_dones.append([[False for x in range(data.max_preds_nn)]])
    model = params["model"]
    use_deparg = params["use_deparg"]
    donotusenopred = int(params["donotusenopred"][0])
    donotusenopred = 0 if donotusenopred==2 else donotusenopred

    current_shape = (data.max_words,data.max_preds_nn)
    deparg_shape  = (data.max_words,data.max_preds_nn+1)
    current_to_predargs = parsemodel.current_to_predargs

    ## validation
    for counttrans in range(data.max_trans):
        countsent=0
        allsent=len(data.golds[task][split])
        logger.info("[valid][%s] Sent: %d"%(split,allsent))
        for iw,il,ip,mask,sent,current_,isfinish,pred_dones in zip(\
                data.inputs_wid[task][split],
                data.inputs_lid[task][split],
                data.inputs_pid[task][split],
                data.masks[task][split],
                data.datasets[task][split],
                currents,
                flags_finish,
                flags_pred_dones\
                ):
            ids_w=iw
            ids_l=il
            ids_p=ip
            if isfinish[0]: continue
            current=current_[0]

            state_t = [ids_w,ids_l,ids_p,None,None,mask,sent,current,None,None,None]
            if use_deparg:
                nninput = state_to_nninput_deparg(state_t, data, task, current_shape, deparg_shape, current_to_predargs)
            else:
                nninput = state_to_nninput(state_t, data, task, current_shape, deparg_shape)
            if not params["debug_dqn_pseudo_Q"]:
                try:
                    scores=model.Q_values(nninput)
                except:
                    import ipdb; ipdb.set_trace()
                    raise
            else:
                scores=model.Q_values_random(nninput)

            argsort=np.argsort(scores)
            countnull=0
            #for i,a in enumerate(argmaxs):
            XPRED=1
            for i,(s,d) in enumerate(zip(argsort,pred_dones[0])):
                if not i<current.shape[0]: break  # pred num limit
                if d:
                    countnull+=1
                    continue
                for arg in reversed(range(scores.shape[1])): # search for non-1 arg as possible
                    a=s[arg]
                    if a==0:
                        countnull+=1          # NULL transition
                        pred_dones[0][i]=True
                        break
                    elif XPRED==2 and a==1:
                        if donotusenopred: continue
                        countnull+=1            # This is not a predicate.
                        pred_dones[0][i]=True
                        break
                    elif a<current.shape[1]+XPRED:            # arg (words) num limit
                        if current[i,a-XPRED]==1:   # already notated, skip
                            continue
                        # no self2self arc
                        # when i==0, it is impossible to predict a root2root arc.
                        elif a-XPRED-1 == i:
                            continue
                        else:
                            current[i,a-XPRED]=1
                            break
            current_[0]=current
            if countnull==current.shape[0]:
                isfinish[0] = 1
            logger.info("[valid][%d]%6.2f%% sent: #%d"%(counttrans,countsent*100./allsent,sent[0]))
            countsent+=1
            if params["debug_one_sentence"]: break
        if checkflag(flags_finish)==allsent:    break
    return currents

def validation_analysis(params, data, task, split):

    ## init datasets
    currents=[]
    flags_finish=[]
    flags_pred_dones=[]
    for z in data.inits[task][split]:
        currents.append([z.copy()])
        flags_finish.append([0])
        flags_pred_dones.append([[False for x in range(data.max_preds_nn)]])
    model = params["model"]
    use_deparg = params["use_deparg"]
    donotusenopred = int(params["donotusenopred"][0])
    donotusenopred = 0 if donotusenopred==2 else donotusenopred

    current_shape = (data.max_words,data.max_preds_nn)
    deparg_shape  = (data.max_words,data.max_preds_nn+1)
    current_to_predargs = parsemodel.current_to_predargs

    interstates_newfaces = []

    allsent=len(data.datasets[task][split])

    ## validation
    countsent=0
    for iw,il,ip,mask,sent,current_,isfinish,pred_dones in zip(\
            data.inputs_wid[task][split],
            data.inputs_lid[task][split],
            data.inputs_pid[task][split],
            data.masks[task][split],
            data.datasets[task][split],
            currents,
            flags_finish,
            flags_pred_dones,
            ):
        ids_w=iw
        ids_l=il
        ids_p=ip
        if isfinish[0]: continue
        current=current_[0]
        interstates_newface=[]

        for counttrans in range(data.max_trans):
            state_t = [ids_w,ids_l,ids_p,None,None,mask,sent,current,None,None,None]
            if use_deparg:
                nninput = state_to_nninput_deparg(state_t, data, task, current_shape, deparg_shape, current_to_predargs)
            else:
                nninput = state_to_nninput(state_t, data, task, current_shape, deparg_shape)
            if not params["debug_dqn_pseudo_Q"]:
                try:
                    scores=model.Q_values(nninput)
                except:
                    import ipdb; ipdb.set_trace()
                    raise
            else:
                scores=model.Q_values_random(nninput)

            XPRED=1
            while True: # preventing loops for DAG
                argsort=np.argsort(scores)
                newface=np.zeros(current.shape)
                newface_arcs,used_trans=[],[]
                countnull=0
                for i,(s,d) in enumerate(zip(argsort,pred_dones[0])):
                    if not i<current.shape[0]: break  # pred num limit
                    if d:
                        countnull+=1
                        used_trans.append(-1)
                        continue
                    for arg in reversed(range(scores.shape[1])): # search for non-1 arg as possible
                        a=s[arg]
                        if a==0:
                            countnull+=1          # NULL transition
                            pred_dones[0][i]=True
                            used_trans.append(a)
                            break
                        elif XPRED==2 and a==1:
                            if donotusenopred: continue
                            countnull+=1            # This is not a predicate.
                            pred_dones[0][i]=True
                            used_trans.append(a)
                            break
                        elif a<current.shape[1]+XPRED:            # arg (words) num limit
                            if current[i,a-XPRED]==1:   # already notated, skip
                                continue
                            # no self2self arc
                            # when i==0, it is impossible to predict a root2root arc.
                            elif a-XPRED-1 == i:
                                continue
                            else:
                                newface[i,a-XPRED]=1
                                newface_arcs.append(i)
                                used_trans.append(a)
                                break

                current_candidate=current+newface
                loops = loop_detecter(current_candidate[:,1:],debug=False)
                if loops==[]:
                    current=current_candidate
                    break

                # if current+newface has loops
                looping_elems = get_uniq(loops)
                #import ipdb; ipdb.set_trace()
                looping_elems = [ x for x in looping_elems if x in newface_arcs] # filter arcs that are newly added
                looping_scores = [scores[e,used_trans[e]] for e in looping_elems]
                smallest_arc = looping_elems[np.argmin(looping_scores)]
                scores[smallest_arc,used_trans[smallest_arc]]=-np.inf

            current_[0]=current
            interstates_newface.append(newface)
            logger.info("[valid][%d]%6.2f%% sent: #%d"%(counttrans,countsent*100./allsent,sent[0]))
            if countnull==current.shape[0]:
                break
        interstates_newfaces.append(interstates_newface)
        countsent+=1
    return currents

def search_loop(mat, path, seen, row):
    loops=[]
    for i in range(mat.shape[0]):
        if mat[i][row]==1:
            if i in path:
                #loops+=1
                print("path",path)
                loops.append(path)
            else:
                if i not in seen:
                    seen.add(i)
                    path_=path[:]
                    path_.append(i)
                    loops+=search_loop(mat,path_,seen,i)
    return loops

def loop_detecter(mat,debug=False):
    # mat[post][pre]
    post,pre=mat.shape
    seen=set()
    loops=[]
    for j in range(post):
        if j not in seen:
            seen.add(j)
            tmp=search_loop(mat,[j],seen,j)
            if debug:
                print(j,tmp,seen)
            loops+=tmp
    if debug:
        print(seen)
        print(loops)
    return loops

def get_uniq(loops):
    looping_elem=set()
    for loop in loops:
        for l in loop:
            looping_elem.add(l)
    return looping_elem

def validation_label(params, data, task, split, where_to_label=None):
    ## init datasets
    labels=[]
    labela=[]
    for z in data.labls[task][split]:
        labels.append(np.ones_like(z)*-1)
        labela.append(np.ones_like(z)*-1)
    nnl = params["model_labeling"]
    if where_to_label is None:
        where_to_label = data.golds[task][split]
    ## validation
    countsent=0
    allsent=len(data.golds[task][split])

    for ids_w,ids_l,ids_p,gold,mask,sent,label,labea in zip(\
            data.inputs_wid[task][split],
            data.inputs_lid[task][split],
            data.inputs_pid[task][split],
            where_to_label,
            data.masks[task][split],
            data.datasets[task][split],
            labels,labela
            ):
        nninput = task, [ids_w,ids_l,ids_p]

        if not params["debug_dqn_pseudo_Q"]:
            try:
                scores=nnl.Q_values(nninput)
            except:
                import ipdb; ipdb.set_trace()
                raise
        else:
            scores=nnl.Q_values_random(nninput)
        argmaxs=np.argmax(scores,axis=2)
        for i,(row,grow) in enumerate(zip(argmaxs,gold)):
            for j,(c,gc) in enumerate(zip(row,grow)):
                labea[i,j]=c
                if gc==0: continue
                label[i,j]=c
        logger.info("[valid][label]%6.2f%% sent: #%d"%(countsent*100./allsent,sent[0]))
        countsent+=1
        if params["debug_one_sentence"]: break
    return labels,labela

def validation_attach_label(labls,golds,preds,labels):
    results=[]
    for z in labels:
        results.append(np.ones_like(z)*-1)
    for rmat,lmat,pmat,gmat,bmat in zip(results,labels,preds,golds,labls):
        try:
            assert gmat.shape==pmat.shape
            assert lmat.shape==pmat.shape
        except:
            import ipdb; ipdb.set_trace()
            print("Shapes mismatch! :"+str(lmat.shape)+", "+str(pmat.shape))
        for i,(lrow,prow) in enumerate(zip(lmat,pmat)):
            for j,(c,pc) in enumerate(zip(lrow,prow)):
                if pc==0: continue
                rmat[i,j]=c
    return results


def checkflag(flags):
    return sum([f[0] for f in flags])

def counttrue(flags):
    return sum([1 if f else 0 for f in flags])
