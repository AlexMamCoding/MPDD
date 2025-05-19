#!/usr/bin/env bash
set -e

# Default Training Parameters
DATA_ROOTPATH="/root/mpdd-test/MPDD-Young"
TRAIN_MODEL="/root/MPDD-main/MPDD-main/checkpoints/1s_2labels_wav2vec+openface/best_model_2025-05-19-04.52.04.pth "
AUDIOFEATURE_METHOD="wav2vec" # Audio feature type, options {wav2vec, opensmile, mfccs}
VIDEOLFEATURE_METHOD="openface" # Video feature type, options {openface, resnet, densenet}
SPLITWINDOW="1s" # Window duration, options {"1s", "5s"}
LABELCOUNT=2 # Number of label categories, options {2, 3, 5}
TRACK_OPTION="Track2"
FEATURE_MAX_LEN=25 # Set maximum feature length; pad with zeros if insufficient, truncate if exceeding. For Track1, options {26, 5}; for Track2, options {25, 5}
BATCH_SIZE=1
DEVICE="cuda"

for arg in "$@"; do
  case $arg in
    --data_rootpath=*) DATA_ROOTPATH="${arg#*=}" ;;
    --train_model=*) TRAIN_MODEL="${arg#*=}" ;;
    --audiofeature_method=*) AUDIOFEATURE_METHOD="${arg#*=}" ;;
    --videofeature_method=*) VIDEOLFEATURE_METHOD="${arg#*=}" ;;
    --splitwindow_time=*) SPLITWINDOW="${arg#*=}" ;;
    --labelcount=*) LABELCOUNT="${arg#*=}" ;;
    --track_option=*) TRACK_OPTION="${arg#*=}" ;;
    --feature_max_len=*) FEATURE_MAX_LEN="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --lr=*) LR="${arg#*=}" ;;
    --num_epochs=*) NUM_EPOCHS="${arg#*=}" ;;
    --device=*) DEVICE="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

for i in `seq 1 1 1`; do
    cmd="python test.py \
        --data_rootpath=$DATA_ROOTPATH \
        --train_model=$TRAIN_MODEL \
        --audiofeature_method=$AUDIOFEATURE_METHOD \
        --videofeature_method=$VIDEOLFEATURE_METHOD \
        --splitwindow_time=$SPLITWINDOW \
        --labelcount=$LABELCOUNT \
        --track_option=$TRACK_OPTION \
        --feature_max_len=$FEATURE_MAX_LEN \
        --batch_size=$BATCH_SIZE \
        --device=$DEVICE"

    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done
