#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import pickle
import logging
import codecs
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import json

SPECIALS=["#EOS","#PAD","#UNK","#EOS_DEP","#PAD_DEP","#NON_DEP","#YEAR","#FLOATDIGIT","#COMMADIGIT","#DIGIT","#NUM_NUM","#NUM_NON","#HYPEN"]
SPECIALS_P=["#EOS","#PAD","#EOS_DEP","#PAD_DEP","#NON_DEP"]

def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(unicode_csv_data,
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        yield row

logger = logging.getLogger(__name__)

class conll_row_id:
    ID = 0
    WORD = 1
    LEMMA = 2
    POS = 3
    CPOS = 3 # We do not use
    TOP = 4
    PREDICATE = 5
    ARGUMENTS = 7

    def dm_form(self):
        self.ID = 0
        self.WORD = 1
        self.LEMMA = 2
        self.POS = 3
        self.CPOS = 3 # We do not use
        self.TOP = 4
        self.PREDICATE = 5
        self.ARGUMENTS = 6

def normalize2(word,idv=None):
    """
    From sillogism implementation (Matthew Honnibal)
    Simple normalization of digit words in the corpus

    # if '-' in word and word[0] != '-':
    #     return '!HYPHEN'
    # elif word.isdigit() and len(word) == 4:
    #     return '!YEAR'
    # elif word[0].isdigit():
    #     return '!DIGITS'
    """
    noncomma=word.replace(",","")
    word=word.replace(",",".")

    if word.isdigit():
        if int(word)>1000 and int(word)<2100:
            return "#YEAR"
        return '#DIGIT'
    try:
        float(word)
        return '#FLOATDIGIT'
    except:
        pass
    try:
        float(noncomma)     # ex. 150,000.000
        return '#COMMADIGIT'
    except:
        pass

    if "-" in word:
        subwords=word.split("-")
        try:
            float(subwords[0])
            try:
                float(subwords[-1])
                return '#NUM_NUM'
            except:
                return '#NUM_NON'
        except:
            #if idv:
            #    if word.lower() not in idv:
            #        return "#HYPHEN"
            pass
    return word.lower()

def normalize(word):
    """
    From sillogism implementation (Matthew Honnibal)
    Simple normalization of digit words in the corpus

    # if '-' in word and word[0] != '-':
    #     return '!HYPHEN'
    # elif word.isdigit() and len(word) == 4:
    #     return '!YEAR'
    # elif word[0].isdigit():
    #     return '!DIGITS'
    """
    noncomma=word.replace(",","")
    word=word.replace(",",".")

    if word.isdigit():
        if int(word)>1000 and int(word)<2100:
            return "#YEAR"
        return '#DIGIT'
    try:
        float(word)
        return '#FLOATDIGIT'
    except:
        pass
    try:
        float(noncomma)     # ex. 150,000.000
        return '#COMMADIGIT'
    except:
        pass

    return word.lower()

def dump_emb(data,file):
    vars=[data.max_words,
     data.max_preds,
     data.max_words_input,
     data.max_words_nn,
     data.max_preds_nn,
     ]
    json.dump([data.widv,data.pidv,data.embidv,data.labl_sets,vars],open(file+".json","w"))
    np.save(file+".wmat",data.wmat,allow_pickle=False)
    np.save(file+".pmat",data.pmat,allow_pickle=False)
    np.save(file+".embmat",data.embmat,allow_pickle=False)

def load_emb(data,file):
    data.widv,data.pidv,data.embidv,data.labl_sets,vars = json.load(open(file+".json","r"))
    data.wmat = np.load(file+".wmat.npy",allow_pickle=False)
    data.pmat = np.load(file+".pmat.npy",allow_pickle=False)
    data.embmat = np.load(file+".embmat.npy",allow_pickle=False)
    data.max_words,data.max_preds,data.max_words_input,data.max_words_nn,data.max_preds_nn=vars
    return data

class Datasets:
    wordcounts=None
    wcounts=None
    lcounts=None
    poscounts=None
    tasks=None
    splits=None
    datasets=None

    # statistics
    max_words=None
    max_preds=None
    max_trans=10
    max_words_nn=None
    max_preds_nn=None

    # expanding dataset
    inputs_wid=None
    inputs_pid=None
    labl_sets=None
    labls=None
    golds=None
    preds_at=None
    inits=None
    masks=None

    # storing ID maps and embeddings
    widv, wmat = None, None
    pidv, pmat = None, None
    embidv, embmat = None, None
    unk, add = None, None

    labl_sets={}

    params=None # for compatibility, not used

    def __init__(self,params=None):
        self.wordcounts=defaultdict(int)
        self.wcounts=defaultdict(int)
        self.lcounts=defaultdict(int)
        self.counts=defaultdict(int)
        self.poscounts=defaultdict(int)
        self.datasets=defaultdict(dict)
        self.params=None
    def read_conll_sem_file(self,input_file, conll_form="", use_external_form=False, mix_lemmma_and_words=True):
        logger.debug('Processing CONLL file :'+input_file)
        form = conll_row_id()
        if conll_form=="dm":
            form.dm_form()
        if self.wcounts is None:
            self.wcounts=defaultdict(int)
            self.lcounts=defaultdict(int)
            self.poscounts=defaultdict(int)
        with codecs.open(input_file,"r","utf8") as csvfile:
            reader = unicode_csv_reader(csvfile, delimiter='\t')
            words,lemmas,poss,tops,preds,pa=[],[],[],[],[],[]
            sentid=-1
            ids=0
            for row in reader:
                if len(row) < 3:    # EOS
                    if len(row)==1:
                        if row[0][0]=="#":
                            sentid=int(row[0][1:])
                    if words==[]: continue
                    assert num_of_pred==len(preds)
                    yield sentid,words,lemmas,poss,tops,preds,pa
                    words,lemmas,poss,tops,preds,pa=[],[],[],[],[],[]
                    ids=0
                else:
                    assert int(row[form.ID])==ids+1
                    word = normalize2(row[form.WORD])
                    self.wcounts[word]+=1
                    words.append(word)
                    lemma=normalize(row[form.LEMMA])
                    if not use_external_form:
                        if mix_lemmma_and_words:
                            self.wcounts[lemma]+=1
                        self.lcounts[lemma]+=1
                    lemmas.append(lemma)
                    pos=row[form.POS]
                    poss.append(pos)
                    self.poscounts[pos]+=1
                    if row[form.TOP]=="+":
                        tops.append(ids)    # 0-base
                    if row[form.PREDICATE]=="+":
                        preds.append(ids)   # 0-base
                    if ids==0:
                        num_of_pred=len(row[form.ARGUMENTS:])
                    else:
                        assert num_of_pred==len(row[form.ARGUMENTS:])
                    pa.append([None if r=="_" else r for r in row[form.ARGUMENTS:]])
                    #import ipdb; ipdb.set_trace()
                    ids+=1
    def merge_word_lemma_counts(self):
        vocab=self.wordcounts
        for wl in [self.wcounts,self.lcounts]:
            for k,v in wl.items():
                if k in vocab:
                    vocab[k]=max(vocab[k],v)
                else:
                    vocab[k]=v
        self.wordcounts=vocab
    def read(self,filepath,task,split,show=[],form=None,mix_lemmma_and_words=False):
        sentnum=0
        self.datasets[task][split]=[]
        if self.unk:
            unkcount={k:0 for k in self.unk}
        for sent in self.read_conll_sem_file(filepath,form,mix_lemmma_and_words=mix_lemmma_and_words):
            if sent[0] in show:
                sentid,words,lemmas,poss,tops,preds,pa=sent
                print(sent[0])
                print(sent[1])
                print(sent[2])
                print(sent[3])
                print("top:",sent[4])
                print("pred:",sent[5])
                #check_multiple(sent[6])
                printpamat(sent)
                #count_pruner(sent[1],sent[7])
            if self.unk:
                sentid,words,lemmas,poss,tops,preds,pa=sent
                for word in words:
                    if word in self.unk:
                        unkcount[word]+=1
            self.datasets[task][split].append(sent)
            sentnum+=1
            if sent[0]==show:
                print()
                break
        if self.unk:
            unks,allunks,topunk=0,0,0
            for k,v in unkcount.items():
                if v!=0:
                    unks+=1
                allunks+=v
                topunk=v if v>topunk else topunk
                if v>2:
                    print(k,v)
            print("%d, %f %%, %d, %d"%(unks,float(unks)/len(self.unk),allunks,topunk))
        print()
        print("Total number of sentences is ",sentnum)

    def fshow(self,f,sent):
        sentid,words,lemmas,poss,tops,preds,pa=sent
        f.write(str(sentid)+"\n")
        f.write(str(words)+"\n")
        f.write(str(lemmas)+"\n")
        f.write(str(poss)+"\n")
        f.write("top:"+str(tops)+", pred:"+str(preds)+"\n")
        #check_multiple(sent[6])
        fprintpamat(f,sent)
        #count_pruner(sent[1],sent[7])

    def show(self,sent):
        sentid,words,lemmas,poss,tops,preds,pa=sent
        print(sentid)
        print(words)
        print(lemmas)
        print(poss)
        print("top:",tops,", pred:",preds)
        #check_multiple(sent[6])
        printpamat(sent)
        #count_pruner(sent[1],sent[7])

    def read_embedding(self,input_file):
        logger.info('Processing embedding file :'+input_file)
        #with open(input_file) as csvfile:
        embdim=None
        embidv={}
        embmat=[]
        with codecs.open(input_file,"r","utf8") as csvfile:
            #reader = csv.reader(csvfile, delimiter='\t')
            reader = unicode_csv_reader(csvfile, delimiter=' ')
            countword=0
            for row in reader:
                try:
                    word=row[0]
                    vec =row[1:]
                    if embdim:
                        assert len(vec)==embdim
                    else:
                        embdim=len(vec)
                    vec=np.array([float(v) for v in vec])
                except:
                    print("embedding dimension mismatch?")
                    import ipdb; ipdb.set_trace()
                if word in embidv:
                    print(word+" is aleady in self.idv...")
                else:
                    embidv[word]=countword
                    embmat.append(vec)
                    countword+=1
        self.embidv=embidv
        self.embmat=np.stack(embmat)
        logger.info('Loaded embedding shape is :'+str(self.embmat.shape))

    def init_nn_inputs(self, use_pred_at=False, pred_select=True):
        self.init_maxs(use_pred_at, pred_select)
        if self.labl_sets=={}:
            self.init_labl_sets()
        self.init_inputs()
        self.init_labls(use_pred_at, pred_select)
        self.init_golds(use_pred_at, pred_select)
        self.init_preds_at()
        self.init_zeros(use_pred_at, pred_select)
        self.init_masks(use_pred_at, pred_select)

    def init_nn_inputs_limited(self, use_pred_at=False, pred_select=True):
        self.init_inputs()
        self.init_zeros(use_pred_at, pred_select)
        self.init_masks(use_pred_at, pred_select)

    ### statistics

    def init_maxs(self, use_pred_at, pred_select):
        self.max_words=0
        self.max_preds=0
        for task in self.tasks:
            for split in self.splits:
                for sent in self.datasets[task][split]:
                    sentid,words,lemmas,poss,tops,preds,pa=sent
                    self.max_words = len(words) if len(words)>self.max_words else self.max_words
                    if use_pred_at:
                        self.max_preds = len(preds) if len(preds)>self.max_preds else self.max_preds
                    else:
                        self.max_preds = self.max_words
        print("self.max_words",self.max_words)
        print("self.max_preds",self.max_preds)
        X=1
        self.max_words_input=self.max_words+1 ## add #EOS (for nn input)
        self.max_words_nn=self.max_words+X    ## add NULL-transition and Non-predicate (for nn prediction)
        self.max_preds_nn=self.max_preds+1    ## add top node

    def init_max_trans(self):
        assert self.golds is not None
        self.max_trans=0
        for task in self.tasks:
            for split in self.splits:
                for gold in self.golds[task][split]:
                    for g in gold:
                        trans_including_top = np.sum(g)
                        self.max_trans = trans_including_top if trans_including_top>self.max_trans else self.max_trans
        print("self.max_trans",self.max_trans)

    ### NN inputs

    def genemb_wo_emb(self,dim,minfreq):
        self.dim = dim
        self.widv, self.wmat = prune_vocab(self.wordcounts,self.dim,SPECIALS,minfreq)
        self.pidv, self.pmat = prune_vocab(self.poscounts,self.dim,SPECIALS_P)

    def genemb_w_emb(self,minfreq):
        self.dim = self.embmat.shape[1]
        self.widv, self.wmat, self.unk, self.add =\
                prune_vocab_w_emb(self.embidv,self.embmat,self.wordcounts,SPECIALS,minfreq)
        self.pidv, self.pmat = prune_vocab(self.poscounts,self.dim,SPECIALS_P)

    def init_inputs(self):
        self.inputs_wid=defaultdict(dict)
        self.inputs_lid=defaultdict(dict)
        self.inputs_pid=defaultdict(dict)
        for task in self.tasks:
            for split in self.splits:
                self.inputs_wid[task][split]=[]
                self.inputs_lid[task][split]=[]
                self.inputs_pid[task][split]=[]
                for sent in self.datasets[task][split]:
                    sentid,words,lemmas,poss,tops,preds,pa=sent
                    # #EOS is added to the last of the arg words.
                    # #EOS is for the convenience of Bi-LSTM inputs.
                    # #EOS is not for the arg-pred analysis.
                    self.inputs_wid[task][split].append(change_word_to_idx(words,  self.widv, self.max_words_input, True))
                    self.inputs_lid[task][split].append(change_word_to_idx(lemmas, self.widv, self.max_words_input, True))
                    self.inputs_pid[task][split].append(change_word_to_idx(poss ,  self.pidv, self.max_words_input, True))

    def init_preds_at(self):
        self.preds_at=defaultdict(dict)
        for task in self.tasks:
            for split in self.splits:
                self.preds_at[task][split]=[]
                for sent in self.datasets[task][split]:
                    sentid,words,lemmas,poss,tops,preds,pa=sent
                    self.preds_at[task][split].append(get_preds_at(preds,self.max_preds, self.max_words))

    # define label classes
    def init_labl_sets(self):
        self.labl_sets={}
        for task in self.tasks:
            split = "train"
            self.labl_sets[task] = initlablset(self.datasets[task][split])

    def init_labls(self, use_pred_at, pred_select):
        self.labls=defaultdict(dict)
        self.labls_mask=defaultdict(dict)
        for task in self.tasks:
            labl_sets = self.labl_sets[task]
            for split in self.splits:
                self.labls[task][split]=[]
                self.labls_mask[task][split]=[]
                for sent in self.datasets[task][split]:
                    sentid,words,lemmas,poss,tops,preds,pa=sent
                    mat,mat_mask=loadlablmat(sent, labl_sets, use_pred_at, pred_select)
                    self.labls[task][split].append(mat)
                    self.labls_mask[task][split].append(mat_mask)

    def init_golds(self, use_pred_at, pred_select):
        self.golds=defaultdict(dict)
        for task in self.tasks:
            for split in self.splits:
                self.golds[task][split]=[]
                for sent in self.datasets[task][split]:
                    self.golds[task][split].append(loadgoldmat(sent, use_pred_at, pred_select))

    def init_zeros(self, use_pred_at, pred_select):
        self.inits=defaultdict(dict)
        for task in self.tasks:
            for split in self.splits:
                self.inits[task][split]=[]
                for sent in self.datasets[task][split]:
                    self.inits[task][split].append(loadinitmat(sent, use_pred_at, pred_select))
    def init_masks(self, use_pred_at, pred_select):
        self.masks=defaultdict(dict)
        for task in self.tasks:
            for split in self.splits:
                self.masks[task][split]=[]
                for sent in self.datasets[task][split]:
                    sentid,words,lemmas,poss,tops,preds,pa=sent

                    X=1
                    # preds+1 , because of TOP(ROOT)-arcs
                    # words+2 , because of NULL-transition and NOT_PRED
                    if use_pred_at:
                        self.masks[task][split].append([\
                                loadmask0( len(preds)+1, len(words)+X, self.max_preds_nn, self.max_words_nn),
                                loadmask1( len(preds)+1, len(words)+X, self.max_preds_nn, self.max_words_nn),
                                loadmask2( len(preds)+1,               self.max_preds_nn),
                                loadmask2( len(preds)  ,               self.max_words_input),
                            ])
                    elif pred_select:
                        self.masks[task][split].append([\
                                loadmask0( len(words)  , len(words)+X+1, self.max_words, self.max_words+X+1),
                                loadmask1( len(words)  , len(words)+X+1, self.max_words, self.max_words+X+1),
                                loadmask2( len(words)  ,                 self.max_words),
                                loadmask2( len(words)  ,                 self.max_words),
                            ])
                    else:
                        self.masks[task][split].append([\
                                loadmask0( len(words)+1, len(words)+X, self.max_preds_nn, self.max_words_nn),
                                loadmask1( len(words)+1, len(words)+X, self.max_preds_nn, self.max_words_nn),
                                loadmask2( len(words)+1,               self.max_preds_nn),
                                loadmask2( len(words)  ,               self.max_words_input),
                            ])

    def save_conll_sem(self,path):
        print("Save to "+path)
        pickle.dump(self,open(path,"wb"))

def change_word_to_idx(words, idv, max_words_nn, addeos):
    r=[]
    for w in words:
        if w in idv:
            r.append(idv[w])
        else:
            r.append(idv["#UNK"])
    if addeos:
        r.append(idv["#EOS"])
        assert max_words_nn-len(words)-1>=0
        r=np.pad(np.array(r),(0,max_words_nn-len(words)-1),'constant',constant_values=(idv["#PAD"]))
    else:
        assert max_words_nn-len(words)>=0
        r=np.pad(np.array(r),(0,max_words_nn-len(words))  ,'constant',constant_values=(idv["#PAD"]))
    return r

def get_preds_at(preds, max_preds, max_words):
    r=np.pad(preds,(0,max_preds-len(preds)),'constant',constant_values=(max_words))
    return r

def prune_vocab_w_emb(idv,embmat,vocab,specials=[],minfreq=0):
    count=embmat.shape[0]
    dim  =embmat.shape[1]
    for i,k in enumerate(specials):
        if k in idv:
            print("SPECIAL SKIP: "+k+" is already in idv...")
        idv[k]=count
        count+=1
    unk,add=set(),set()
    for k,v in vocab.items(): # vocab : {"word", freq}
        if k in idv:
            #print("SKIP: "+k+" is already in idv...")
            pass
        elif v<minfreq:
            print("UNK: ",k,v)
            unk.add(k)
        else:
            print("ADD: ",k,v)
            add.add(k)
            idv[k]=count
            count+=1
    embmat_expand = np.random.rand(len(idv)-embmat.shape[0],dim)*0.3
    print("Expanded from %d to %d ."%(embmat.shape[0],len(idv)))
    e2=np.vstack([embmat,embmat_expand])
    return idv,e2,unk,add

def prune_vocab(vocab,dim,specials=[],minfreq=0):
    idv={}
    count=0
    for i,k in enumerate(specials):
        idv[k]=count
        count+=1
    for k,v in vocab.items(): # vocab : {"word", freq}
        if k in idv:
            print("SKIP: "+k+" is already in idv...")
        elif v<minfreq:
            print("UNK: ",k,v)
        else:
            idv[k]=count
            count+=1
    embmat = np.random.rand(len(idv),dim)*0.1
    print("Shrinked from %d to %d ."%(len(vocab),len(idv)))
    return idv,embmat

def read_raw_file(input_file):
    logger.debug('Processing UTF-8 sentence file ... ')
    sentences = []
    with codecs.open(input_file,"r","utf8") as reader:
        #reader = unicode_csv_reader(csvfile, delimiter='\t')
        sentence = []

        #for row,i in zip(reader,range(10)):
        for i,row in enumerate(reader):
            row=row.rstrip()            #remove all space and LF
            #print row
            if len(row)==0: continue
            adds=[]
            for char in row:
                #print char,type(char),
                add=(normalize(char),None,None,None,1,normalize(char),char)
                adds.append(add)
            sentences.append(adds)
            #if i==9: break
    return sentences

def read_raw_file_generator(input_file):
    logger.debug('Processing UTF-8 sentence file ... ')
    sentences = []
    with codecs.open(input_file,"r","utf8") as reader:
        #reader = unicode_csv_reader(csvfile, delimiter='\t')
        sentence = []

        #for row,i in zip(reader,range(10)):
        for i,row in enumerate(reader):
            row=row.rstrip()            #remove all space and LF
            #print row
            if len(row)==0: continue
            adds=[]
            for char in row:
                #print char,type(char),
                add=(normalize(char),None,None,None,1,normalize(char),char)
                adds.append(add)
            yield adds
            #if i==9: break


def check_multiple(pa):
    for row in pa:
        #print(row)
        if sum([0 if r is None else 1 for r in row])>=2:
            print(["_" if r is None else r for r in row])


PRUNER_SYMBOLS_old=[":",";",".","?","(",")","{","}","”","“","‘","’","-","–"]
PRUNER_SYMBOLS_20180508=[".","?","{","“","‘","-"]
PRUNER_SYMBOLS=[]
#PRUNER_SYMBOLS=[";","(",")","}","”","’","–"]

def fprintpamat(f,sent):
    sentid,words,lemmas,poss,tops,preds,pa=sent
    for w,row in zip(words,pa):
        if w in PRUNER_SYMBOLS: continue
        for r in row:
            if r is None:
                f.write("0")
            else:
                f.write("1")
        f.write("\t")
        for r in row:
            if r is not None:
                f.write(r+"\t")
        f.write("\n")

def printpamat(sent):
    sentid,words,lemmas,poss,tops,preds,pa=sent
    for w,row in zip(words,pa):
        if w in PRUNER_SYMBOLS: continue
        for r in row:
            if r is None:
                print(0,end="")
            else:
                print(1,end="")
        print("\t",end="")
        for r in row:
            if r is not None:
                print(r+"\t",end="")
        print()

def initlablset(sents):
    labels = {"TOP":0}
    label_count=1
    for sent in sents:
        sentid,words,lemmas,poss,tops,preds,pa=sent
        mat=[]
        for i,(w,row) in enumerate(zip(words,pa)):
            for r in row:
                assert r!="TOP"
                if r not in labels:
                    labels[r]=label_count
                    label_count+=1
        #import ipdb; ipdb.set_trace()
    return labels

def loadlablmat(sent, labl_sets, use_pred_at=False, pred_select=False):
    sentid,words,lemmas,poss,tops,preds,pa=sent
    mat=[]
    mat_mask=[]
    for i,(w,row) in enumerate(zip(words,pa)):
        m=[]
        m_mask=[]
        if w in PRUNER_SYMBOLS: continue
        # top arcs
        if i in tops:
            m.append(0)
        else:
            m.append(-1)
        m_mask.append(1)

        # assigning argument - predicate relations
        if use_pred_at:
            for r in row:
                if r is None:
                    m.append(-1)
                else:
                    m.append(labl_sets[r])
                m_mask.append(1)
        else:
            pred_num=0
            for i in range(len(words)):
                if i in preds:
                    r=row[pred_num]
                    pred_num+=1
                    if r is None:
                        m.append(-1)
                        m_mask.append(1)
                    elif r in labl_sets:    ## TODO
                        m.append(labl_sets[r])
                        m_mask.append(1)
                    else:
                        #print("label "+r+" is missing!")
                        m.append(-1)
                        m_mask.append(0)
                else:
                    m.append(-1)
                    m_mask.append(1)
            assert pred_num == len(preds)
        mat.append(m)
        mat_mask.append(m_mask)
    mat = np.array(mat)
    mat_mask = np.array(mat_mask)
    assert mat.shape==mat_mask.shape
    if not use_pred_at:
        assert mat.shape[0]+1 == mat.shape[1]
    #print(mat)
    #import ipdb; ipdb.set_trace()
    if not pred_select:
        mat = mat.transpose()
        mat_mask = mat_mask.transpose()
    return mat,mat_mask

def loadgoldmat(sent, use_pred_at=False, pred_select=False):
    sentid,words,lemmas,poss,tops,preds,pa=sent
    mat=[]
    for i,(w,row) in enumerate(zip(words,pa)):
        m=[]
        if w in PRUNER_SYMBOLS: continue
        # top arcs
        if i in tops:
            m.append(1)
        else:
            m.append(0)

        # assigning argument - predicate relations
        if use_pred_at:
            for r in row:
                if r is None:
                    m.append(0)
                else:
                    m.append(1)
        else:
            pred_num=0
            for i in range(len(words)):
                if i in preds:
                    r=row[pred_num]
                    pred_num+=1
                    if r is None:
                        m.append(0)
                    else:
                        m.append(1)
                else:
                    m.append(0)
            assert pred_num == len(preds)
        mat.append(m)
    mat = np.array(mat)
    if not use_pred_at:
        assert mat.shape[0]+1 == mat.shape[1]
    #print(mat)
    #import ipdb; ipdb.set_trace()
    if not pred_select:
        mat = mat.transpose()
    return mat

def loadinitmat(sent, use_pred_at=False, pred_select=False):
    sentid,words,lemmas,poss,tops,preds,pa=sent
    mat=[]
    for w,row in zip(words,pa):
        if w in PRUNER_SYMBOLS: continue
        if use_pred_at:
            mat.append([0 for x in range(len(preds)+1)])
        else:
            mat.append([0 for x in range(len(words)+1)])
    mat = np.array(mat)
    #print(mat)
    #import ipdb; ipdb.set_trace()
    if not pred_select:
        mat = mat.transpose()
    return mat

def loadmask0(num_preds, num_words, max_preds_nn, max_words_nn):
    P,N=1,0
    vec=[P  for x in range(num_words)]+[N for x in range(max_words_nn-num_words)]
    neg=[N for x in range(max_words_nn)]
    mat=[vec   for x in range(num_preds)]+[neg   for x in range(max_preds_nn-num_preds)]
    return np.array(mat,dtype=np.float32)

def loadmask1(num_preds, num_words, max_preds_nn, max_words_nn):
    P,N=0,-1e8
    vec=[P  for x in range(num_words)]+[N for x in range(max_words_nn-num_words)]
    neg=[N for x in range(max_words_nn)]
    mat=[vec   for x in range(num_preds)]+[neg   for x in range(max_preds_nn-num_preds)]
    return np.array(mat,dtype=np.float32)

def loadmask2(num_preds, max_preds_nn):
    P,N=1,0
    vec=[P  for x in range(num_preds)]+[N for x in range(max_preds_nn-num_preds)]
    return np.array(vec,dtype=np.float32)

words_pruner=defaultdict(int)
words_count=defaultdict(int)
def count_pruner(words,pa):
    assert len(words)==len(pa)
    for w,row in zip(words,pa):
        words_count[w]+=1
        words_pruner[w]+=sum([0 if r is None else 1 for r in row]) # number of dep_to

## Checking and preprocessing data ##

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument('--file', help='Train/Parse text file (CONLL format)', nargs="+")
    parser.add_argument('--tasks', help='dm/pas/psd', nargs='+', default=["dm"])
    parser.add_argument('--splits', help='train/dev/test', nargs='+', default=["train","dev","test"])

    parser.add_argument('--debug', help='', action="store_true")
    parser.add_argument('--show', help='', nargs=1, default=["21669015"])
    parser.add_argument('--emb', help='Specify an embeddings file', nargs=1, default=[None])
    parser.add_argument('--vocab_min_freq', help='', nargs=1, default=[3])
    parser.add_argument('--datapkl', help='File name to save dataset as pkl', nargs=1, default=["alldata.pkl"])
    parser.add_argument('--datanpy', help='File name to save emdedding data as npy and json', nargs=1, default=["embdata"])
    parser.add_argument('--mix_lemmma_and_words', help='Mixed dict of word and lemma', action="store_true")

    parser.add_argument('--datapkl_to_datanpy', help='Convert datapkl to datanpy', action="store_true")

    opts = parser.parse_args()
    params = OrderedDict(vars(opts))

    if params['datapkl_to_datanpy']:
        path = params["datapkl"][0]
        logger.info("Load from "+path)
        data = pickle.load(open(path,"rb"))
        data.init_maxs(False, True)
        data.init_labl_sets()
        dump_emb(data,params["datanpy"][0])
        quit()

    data = Datasets(None)
    #data.tasks=["dm","pas","psd"]
    data.tasks=params["tasks"][:]
    #data.splits=["train","dev"]
    data.splits=params["splits"][:]

    DEBUG=params["debug"]
    params["mix_lemmma_and_words"]=True

    assert len(data.tasks)*len(data.splits)==len(params["file"])

    if params["emb"][0]:
        data.read_embedding(params["emb"][0])

    def fileindex(task,split,splits):
        return len(splits)*data.tasks.index(task)+splits.index(split)

    for task in data.tasks:
        for split in data.splits:
            file = params["file"][fileindex(task,split,data.splits)]
            print(file)
            if DEBUG:
                data.read(file,task,split,[int(x) for x in params["show"]],mix_lemmma_and_words=params["mix_lemmma_and_words"])
            else:
                data.read(file,task,split,mix_lemmma_and_words=params["mix_lemmma_and_words"])
            if split!="train": continue
            data.merge_word_lemma_counts()

    minfreq=int(params["vocab_min_freq"][0])
    if params["emb"][0]:
        data.genemb_w_emb(minfreq)
    else:
        dim=256
        data.genemb_wo_emb(dim,minfreq)

    if not DEBUG:
        data.save_conll_sem(params["dataset"][0])
        data.init_labl_sets()
        dump_emb(data,params["datanpy"][0])

