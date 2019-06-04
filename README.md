# semrl
Semantic dependency parser with reinforcement learning.

## Requirements
Tensorflow

## Usage
### Parsing
*We will publish off-the-shelve models soon.*

### Trainging
#### Requirements
- This requires the semantic dependency parsing datasets:
  - https://catalog.ldc.upenn.edu/LDC2016T10
- The model also needs pretrained word mebedding vectors such as GloVe.
  - http://nlp.stanford.edu/data/glove.6B.zip
#### Preprocessing data
Firstly you need to prepare your datasets with embedding vectors as `data.pkl`.
```
python3 src/loaddata.py \
--tasks dm pas psd \
--splits train dev \
--datapkl data.pkl \
--vocab_min_freq 7 \
--emb path_to_glove_embedding_txt \
--file \
path_to_dm_train \
path_to_dm_dev \
path_to_pas_train \
path_to_pas_dev \
path_to_psd_train \
path_to_psd_dev \
```
#### Train & Dev
This is an example of training.
```
python src/semrl.py \
    --train --parsing \
    --tasks dm \
    --inittasks dm pas psd \
    --splits train dev \
    --datapkl data.pkl \
    --lstm_layers 3 \
    --lstm_layers_dep 1 \
    --fnn_hidden_dim 4000 \
    --h_dim 600 --h_dep_dim 200 --emb_dep 100 \
    --flag_dim 128 \
    --droppout_rate_fnn 0.5 \
    --epoch_max 40 \
    --use_lemma \
    --use_fnn3 \
    --use_highway_fnn \
    --use_highway_fnn2 \
    --donotusenopred 3 \
    --gold_easy_fast 0 \
    --savepath path_to_save_model \
```
(To be updated soon for details and the labeling model...)

## Updated soon
- Detailed Usage for training.
- Usage for parsing text files of CONLL format.
- Off-the-shelf parsing models for DM, PAS and PSD formalisms.
