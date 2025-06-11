export PYTHONPATH=.
DEVICE=0;
CONFIG="configs/exp/durflex_evc.yaml";
MODEL_NAME="hbrt";
CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
    --config $CONFIG \
    --exp_name $MODEL_NAME \
    --reset