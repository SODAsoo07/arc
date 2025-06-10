export PYTHONPATH=.
DEVICE=0;
CONFIG="configs/exp/durflex_evc.yaml";
python data_gen/runs/preprocess.py --config $CONFIG
python data_gen/runs/binarize.py --config $CONFIG
python preprocess_unit.py --config $CONFIG
