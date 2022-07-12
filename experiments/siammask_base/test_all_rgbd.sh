show_help() {
cat << EOF
Usage:
    ${0##*/} [-h/--help] [-s/--start] [-e/--end] [-d/--dataset] [-m/--model]  [-g/--gpu]
    e.g.
        bash ${0##*/} -s 1 -e 20 -d VOT2021RGBD -g 4 # for test models
        bash ${0##*/} -m snapshot/checkpoint_e10.pth -n 8 -g 4 # for tune models
EOF
}

ROOT=`git rev-parse --show-toplevel`
source activate siammask
export PYTHONPATH=$ROOT:$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

dataset=VOT2021RGBD # VOT2018
NUM=4
START=1
END=20
GPU=0

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit
            ;;
        -d|--dataset)
            dataset=$2
            shift 2
            ;;
        -n|--num)
            NUM=$2
            shift 2
            ;;
        -s|--start)
            START=$2
            shift 2
            ;;
        -e|--end)
            END=$2
            shift 2
            ;;
        -m|--model)
            model=$2
            shift 2
            ;;
        -g|--gpu)
            GPU=$2
            shift 2
            ;;
        *)
            echo invalid arg [$1]
            show_help
            exit 1
            ;;
    esac
done

set -e

if [ -z "$model" ]; then
    echo test snapshot $START ~ $END on dataset $dataset with $GPU gpus.
    for i in $(seq $START $END)
    do
        bash test_rgbd.sh snapshot_RGB/checkpoint_e$i.pth $dataset $(($i % $GPU)) &
    done
    wait

    # calculate the accuracy
    # python $ROOT/tools/eval.py --dataset $dataset --num 20 --tracker_prefix C --result_dir ./test/$dataset 2>&1 | tee logs/eval_test_$dataset.log

    # use vot toolkit
    # vot analysis --workspace /home/sgn/Data1/yan/ ...
else
    echo tuning $model on dataset $dataset with $NUM jobs in $GPU gpus.
    for i in $(seq 1 $NUM)
    do
        bash tune_rgbd.sh $model $dataset $(($i % $GPU)) &
    done
    wait
    rm finish.flag

    # calculate the accuracy
    # python $ROOT/tools/eval.py --dataset $dataset --num 20 --tracker_prefix C  --result_dir ./result/$dataset 2>&1 | tee logs/eval_tune_$dataset.log
fi
