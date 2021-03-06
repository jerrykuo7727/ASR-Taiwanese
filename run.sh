#!/bin/bash

# Copyright 2019 Nagoya University (Masao Someki)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

python3_cmd=python3.6

# general configuration
backend=pytorch
stage=0        # start from -1 if you need to start from data download
stop_stage=0
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
export CUDA_VISIBLE_DEVICES=0
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

# exp tag
tag="" # tag for managing experiments.

train_config=conf/train.yaml
decode_config=conf/decode.yaml
preprocess_config=conf/specaug.yaml
lm_config=conf/lm.yaml

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# legacy setup
data_type=legacy
train_set=train_trim_sp
train_dev=dev_trim
recog_set="dev test"

#adaptation setup
#data_type=speaker-adaptation
#train_set=train_adapt_trim_sp
#train_dev=dev_adapt_trim
#recog_set="dev_adapt test_adapt"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "#################################"
    echo "    Stage 0: Data Preparation    "
    echo "#################################"
    echo `date`
    rm data dump exp fbank -rf
    ${python3_cmd} local/prepare_data.py
    ${python3_cmd} local/preprocess_data.py

    for dset in dev test train; do
        dset_data=`ls -d data/all_split/*_${dset}`
        utils/combine_data.sh data/$dset ${dset_data[*]}

        utils/utt2spk_to_spk2utt.pl data/$dset/utt2spk > data/$dset/spk2utt
        utils/fix_data_dir.sh data/$dset
        mv data/$dset data/$dset.orig
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
    done
    echo `date`
fi


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "###################################"
    echo "    Stage 1: Feature Generation    "
    echo "###################################"
    echo `date`
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in test dev train; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 64 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # remove utt having > 2000 frames or < 10 frames or
    # remove utt having > 400 characters or 0 characters
    remove_longshortdata.sh --maxchars 400 data/train data/train_trim
    remove_longshortdata.sh --maxchars 400 data/dev data/${train_dev}

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train_trim data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train_trim data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train_trim data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 64 --write_utt2num_frames true \
        data/${train_set} exp/make_fbank/${train_set} ${fbankdir}
    utils/fix_data_dir.sh data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
    echo `date`
fi


dict=data/lang_char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "#####################################################"
    echo "    Stage 2: Dictionary and JSON Data Preparation    "
    echo "#####################################################"
    echo `date`
    mkdir -p data/lang_char/

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json labels"
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
    echo `date`
fi


# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "###############################"
    echo "    stage 3: LM Preparation    "
    echo "###############################"
    echo `date`
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
        > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " \
        > ${lmdatadir}/valid.txt
    text2token.py -s 1 -n 1 data/test/text | cut -f 2- -d" " \
        > ${lmdatadir}/test.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
    echo `date`
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then 
	expname=${expname}_$(basename ${preprocess_config%.*}) 
    fi 
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "#################################"
    echo "    Stage 4: Network Training    "
    echo "#################################"
    echo `date`
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --preprocess-conf ${preprocess_config} \
        --config ${train_config} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
    echo `date`
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "#########################"
    echo "    Stage 5: Decoding    "
    echo "#########################"
    echo `date`
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose 1 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        score_sclite.sh ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished decoding and scoring."
    echo `date`
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "##############################"
    echo "    Stage 6: ROUGE Scoring    "
    echo "##############################"
    echo `date`
    for rtask in ${recog_set}; do
        echo "-------- $rtask set --------"
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        ${python3_cmd} local/score_rouge.py ${expdir}/${decode_dir}/data.json
    done
    echo "Finished scoring ROUGE metrics."
    echo `date`
fi
