if [ -z "$4" ]
  then
    echo "Need input parameter!"
    echo "Usage: bash `basename "$0"` \$CONFIG \$MODEL \$DATASET \$GPUID"
    exit
fi

ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

config=$1
model=$2
dataset=$3
gpu=$4

CUDA_VISIBLE_DEVICES=$gpu python -u $ROOT/tools/test_rgbd.py \
    --config $config \
    --resume $model \
    --mask --refine --visualization \
    --dataset $dataset 2>&1 | tee logs/test_$dataset.log
