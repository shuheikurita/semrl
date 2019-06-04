#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gen_minibatches import *
from util_std import *
import numpy as np

import logging
logger = logging.getLogger(__name__)

def get_random_index_of_binary_vector(binary_vec):
    return np.nonzero(binary_vec)[0][np.random.randint(np.sum(binary_vec))]

def current_to_predargs(current,ind_w,deparg_shape,inv,addeos):
    max_preds_nn,max_words_input = deparg_shape
    assert addeos==True
    ans=[]
    id_pad = inv["#PAD_DEP"]
    id_non = inv["#NON_DEP"]
    id_eos = inv["#EOS_DEP"]
    base=np.ones((max_words_input,),dtype=np.int64)*id_pad
    for c_row in current:
        add = [ w if c==1 else id_non for c,w in zip(c_row,ind_w)]+[id_eos]+[id_pad for _ in range(0,max_words_input-len(c_row)-1)]
        ans.append(add)
    for a in range(max_preds_nn-current.shape[0]):
        ans.append(base)
    ans=np.array(ans,dtype=np.int64)
    return ans

def sample_gold_once(gold, current, y_shape, mask_pred, pred_dones, preds_at, donotusenopred, gold_easy_fast): # mask_pred is msk2
    max_preds_nn,max_words_nn = y_shape
    try:
        assert gold.shape==current.shape
    except:
        import ipdb;ipdb.set_trace()
    if np.array_equal(gold,current): return None,current,mask_pred,pred_dones
    ans=[]
    res=gold-current
    next_current=np.copy(current)
    base=np.zeros((max_words_nn,),dtype=np.int64)
    XPRED = 1
    res_ind=np.arange(0,res.shape[1])
    for i,(g,c,r,n,m,d) in enumerate(zip(gold,current,res,next_current,mask_pred,pred_dones)):
        if np.array_equal(g,c):
            add=np.copy(base)
            # i==0 is a top-arc
            if i==0:
                add[0]=1
            #elif i-1 not in preds_at and (not donotusenopred==1):
            elif i-1 not in preds_at:
                add[XPRED-1]=1
                if donotusenopred==2:
                    ans.append(add) # no more arcs
                    continue
            else:
                add[0]=1
            ans.append(add) # no more arcs
            if d and m!=0. and donotusenopred!=3:
                mask_pred[i]=0.
            else:
                pred_dones[i]=True
        else:
            if gold_easy_fast==1:
                ind=np.argmin(np.abs(res_ind-i)+(1-r)*1024)
            elif gold_easy_fast==2:
                to_argmin=np.abs(res_ind-i)+(1-r)*1024
                ind=np.argmin(to_argmin[::-1])
                ind=r.shape[0]-ind-1
            elif gold_easy_fast==3:
                ind=np.nonzero(r)[0]
            else:
                ind=get_random_index_of_binary_vector(r)
            add=np.copy(base)
            add[ind+XPRED]=1
            n[ind]=1
            ans.append(add)
    base[0]=1
    for a in range(max_preds_nn-gold.shape[0]):
        ans.append(base)
    ans=np.array(ans)
    return ans,next_current,mask_pred,pred_dones

def sample_gold_square(gold,current,max_words_nn,max_preds_nn,mask_pred,pred_dones): # mask_pred is msk2
    try:
        assert gold.shape==current.shape
    except:
        import ipdb;ipdb.set_trace()
    if np.array_equal(gold,current): return None,current,mask_pred,pred_dones
    ans=[]
    res=gold-current
    next_current=np.copy(current)
    base=np.zeros((max_words_nn,),dtype=np.int64)
    #print("res",res)
    for i,(g,c,r,n,m,d) in enumerate(zip(gold,current,res,next_current,mask_pred,pred_dones)):
        if np.array_equal(g,c):
            add=np.copy(base)
            add[0]=1
            #add[0]=1
            if d and m!=0.:
                mask_pred[i]=0.
            else:
                pred_dones[i]=True
            ans.append(add) # no more arcs

        else:
            add=np.copy(base)
            add[1:1+gold.shape[1]]=r
            ind=get_random_index_of_binary_vector(r)
            n[ind]=1
            ans.append(add)
    for a in range(max_preds_nn-gold.shape[0]):
        ans.append(base)
    ans=np.array(ans)
    #print("ans",ans,ans.shape)
    #import ipdb;ipdb.set_trace()
    return ans,next_current,mask_pred,pred_dones

def prf_numpy_class(golds,preds,msks=None,ignore_class=-1):
    if type(msks)==type(None):
        sumpred=0
        sumgold=0
        sumandv=0
        sumallv=0
        for gmat,pmat in zip(golds,preds):
            #assert gmat.shape==pmat.shape
            #print(gmat,pmat)
            for gvec,pvec in zip(gmat,pmat):
                for g,p in zip(gvec,pvec):
                    if g==ignore_class: continue
                    if g==p:
                        sumandv+=1
                    sumgold+=1
        #precision=divwoe(sumandv,sumpred)
        recall   =divwoe(sumandv,sumgold)
        logger.info("TP,sumgold = %d,%d"%(sumandv,sumgold))
        return recall
    else:
        raise NotImplemented

def prf_numpy_LAS_macro(golds,preds,msks=None,ignore_class=-1):
    if type(msks)==type(None):
        ps,rs,fs=[],[],[]
        for gmat,pmat in zip(golds,preds):
            try:
                assert gmat.shape==pmat.shape
            except:
                import ipdb; ipdb.set_trace()
                print("Shapes mismatch! :"+str(gmat.shape)+", "+str(pmat.shape))
            sumpred=0
            sumgold=0
            sumandv=0
            sumallv=0
            sumallv+=np.sum(np.ones_like(gmat))
            for gvec,pvec in zip(gmat,pmat):
                sumgold+=sum([1 if g!=-1 else 0 for g in gvec])
                sumpred+=sum([1 if p!=-1 else 0 for p in pvec])
                sumandv+=sum([1 if g==p and g!=-1  else 0 for g,p in zip(gvec,pvec)])
            #assert gmat.shape==pmat.shape
            #print(gmat,pmat)
            precision=divwoe(sumandv,sumpred)
            recall   =divwoe(sumandv,sumgold)
        #logger.info("TP,sumgold = %d,%d"%(sumandv,sumgold))
        #return recall
            ps.append(precision)
            rs.append(recall)
            fs.append(divwoe(2.*(precision*recall),(precision+recall)))
        #logger.info("TP,sumpred,sumgold,all(ref.) = %d,%d,%d,%d"%(sumandv,sumpred,sumgold,sumallv))
        return np.mean(fs),np.mean(ps),np.mean(rs)
    else:
        raise NotImplemented

def prf_numpy_LAS_micro(gmat,pmat,msks=None):
    sumpred=0
    sumgold=0
    sumandv=0
    sumallv=0
    if type(msks)==type(None):
        try:
            assert gmat.shape==pmat.shape
        except:
            import ipdb; ipdb.set_trace()
            print("Shapes mismatch! :"+str(gmat.shape)+", "+str(pmat.shape))
        sumallv+=np.sum(np.ones_like(gmat))
        for gvec,pvec in zip(gmat,pmat):
            sumgold+=sum([1 if g!=-1 else 0 for g in gvec])
            sumpred+=sum([1 if p!=-1 else 0 for p in pvec])
            sumandv+=sum([1 if g==p and g!=-1  else 0 for g,p in zip(gvec,pvec)])
    else:
        raise NotImplemented
        #assert gmat.shape==pmat.shape
        #print(gmat,pmat)
        sumallv+=np.sum(np.ones_like(mmat))
        for gvec,pvec,mvec in zip(gmat,pmat,mmat):
            sumgold+=sum([1 if g!=-1 and m==1 else 0 for g,m in zip(gvec,mvec)])
            sumpred+=sum([1 if p!=-1 and m==1 else 0 for p,m in zip(pvec,mvec)])
            sumandv+=sum([1 if g==p  and m==1 else 0 for g,p,m in zip(gvec,pvec,mvec)])
    precision=divwoe(sumandv,sumpred)
    recall   =divwoe(sumandv,sumgold)
    #logger.info("TP,sumpred,sumgold,all(ref.) = %d,%d,%d,%d"%(sumandv,sumpred,sumgold,sumallv))
    return divwoe(2.*(precision*recall),(precision+recall)),precision,recall

def prf_numpy_LAS(golds,preds,msks=None):
    sumpred=0
    sumgold=0
    sumandv=0
    sumallv=0
    if msks is None:
        for gmat,pmat in zip(golds,preds):
            try:
                assert gmat.shape==pmat.shape
            except:
                import ipdb; ipdb.set_trace()
                print("Shapes mismatch! :"+str(gmat.shape)+", "+str(pmat.shape))
            sumallv+=np.sum(np.ones_like(gmat))
            for gvec,pvec in zip(gmat,pmat):
                sumgold+=sum([1 if g!=-1 else 0 for g in gvec])
                sumpred+=sum([1 if p!=-1 else 0 for p in pvec])
                sumandv+=sum([1 if g==p and g!=-1  else 0 for g,p in zip(gvec,pvec)])
    else:
        print("%d, %d, %d"%(len(golds),len(preds),len(msks)))
        for gmat,pmat,mmat in zip(golds,preds,msks):
            #print(gmat.shape)
            #print(pmat.shape)
            #print(mmat.shape)
            #assert gmat.shape==pmat.shape
            try:
                assert gmat.shape==pmat.shape
                assert gmat.shape==mmat.shape
            except:
                print(gmat.shape)
                print(pmat.shape)
                print(mmat.shape)
                print()
            #print(gmat,pmat)
            sumallv+=np.sum(np.ones_like(gmat))
            for gvec,pvec,mvec in zip(gmat,pmat,mmat):
                sumgold+=sum([1 if g!=-1 or m==0 else 0 for g,m in zip(gvec,mvec)])
                sumpred+=sum([1 if p!=-1 or m==0 else 0 for p,m in zip(pvec,mvec)])
                sumandv+=sum([1 if (g==p and g!=-1) or m==0 else 0 for g,p,m in zip(gvec,pvec,mvec)])
    precision=divwoe(sumandv,sumpred)
    recall   =divwoe(sumandv,sumgold)
    logger.info("TP,sumpred,sumgold,all(ref.) = %d,%d,%d,%d"%(sumandv,sumpred,sumgold,sumallv))
    return divwoe(2.*(precision*recall),(precision+recall)),precision,recall

def prf_numpy2(golds,preds,msks=None):
    if type(msks)==type(None):
        sumpred=0
        sumgold=0
        sumandv=0
        sumallv=0
        for gold,pred in zip(golds,preds):
            #gold=gold_[0]
            #pred=pred_[0]
            assert gold.shape==pred.shape
            andv=gold*pred
            sumpred+=np.sum(pred)
            sumgold+=np.sum(gold)
            sumandv+=np.sum(andv)
            sumallv+=np.sum(np.ones_like(gold))
        precision=divwoe(sumandv,sumpred)
        recall   =divwoe(sumandv,sumgold)
        logger.info("TP,sumpred,sumgold,all(ref.) = %d,%d,%d,%d"%(sumandv,sumpred,sumgold,sumallv))
        return divwoe(2.*(precision*recall),(precision+recall)),precision,recall
    else:
        raise NotImplemented

def prf_numpy2_macro(golds,preds,msks=None):
    if type(msks)==type(None):
        ps,rs,fs=[],[],[]
        for gold,pred in zip(golds,preds):
            sumpred=0
            sumgold=0
            sumandv=0
            sumallv=0
            #gold=gold_[0]
            #pred=pred_[0]
            assert gold.shape==pred.shape
            andv=gold*pred
            sumpred+=np.sum(pred)
            sumgold+=np.sum(gold)
            sumandv+=np.sum(andv)
            sumallv+=np.sum(np.ones_like(gold))
            precision = divwoe(sumandv,sumpred)
            recall    = divwoe(sumandv,sumgold)
            ps.append(precision)
            rs.append(recall)
            fs.append(divwoe(2.*(precision*recall),(precision+recall)))
        #logger.info("TP,sumpred,sumgold,all(ref.) = %d,%d,%d,%d"%(sumandv,sumpred,sumgold,sumallv))
        return np.mean(fs),np.mean(ps),np.mean(rs)
    else:
        raise NotImplemented

def prf_numpy(pred,gold,msk=None):
    if type(msk)==type(None):
        andv=gold*pred
        precision=divwoe(np.sum(andv),np.sum(pred))
        recall   =divwoe(np.sum(andv),np.sum(gold))
        allv=gold*0+1
        print("TP,pred,correct,all = ",np.sum(andv),np.sum(pred),np.sum(gold),np.sum(allv))
        return pred,precision,recall,divwoe(2.*(precision*recall),(precision+recall))
    else:
        predm=np.maximum(pred-msk,0)
        goldm=np.maximum(gold-msk,0)
        andv=goldm*predm
        precision=divwoe(np.sum(andv),np.sum(predm))
        recall   =divwoe(np.sum(andv),np.sum(goldm))
        allv=(gold*0+1)-msk
        print("TP,pred,correct,all,msk = ",np.sum(andv),np.sum(predm),np.sum(goldm),np.sum(allv),np.sum(msk))
        return predm,precision,recall,divwoe(2.*(precision*recall),(precision+recall))

