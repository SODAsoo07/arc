export PYTHONPATH=.
DEVICE=0;
EXP_NAME="./Save_ckpt/DurFlex"
CONFIG="./configs/exp/durflex_evc.yaml";
SRC_WAV="./sample/Sample_Jp_Netural.wav";
SAVE_DIR="./results"
CUDA_VISIBLE_DEVICES=$DEVICE python infer.py --config $CONFIG \
    --exp_name $EXP_NAME \
    --src_wav $SRC_WAV \
    --save_dir $SAVE_DIR