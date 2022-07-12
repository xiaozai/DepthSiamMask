ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siammask_rgbd.py \
    --arch Custom_RGBD \
    --save_dir snapshot_RGBD \
    --config=config.json -b 64 \
    -j 20\
    --epochs 20 \
    --resume snapshot_RGBD/checkpoint_e10.pth \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

# bash test_all_rgbd.sh -s 1 -e 20 -d VOT2021RGBD -g 2 # 4
