set -x
set -e
# ./scripts/train.sh 0 18 synthtext
export CUDA_VISIBLE_DEVICES=$1
IMG_PER_GPU=$2 # 每张卡的batch
DATASET=$3 # 传入的数据集路径名称

# 预训练模型的存放位置
CHKPT_PATH=${HOME}/models/seglink/seglink_synthtext
# 存放训练好的模型的位置
TRAIN_DIR=${HOME}/models/seglink/seglink_icdar2015_384
#TRAIN_DIR=${HOME}/temp/no-use/seglink/seglink_icdar2015_384
#CHKPT_PATH=${HOME}/models/ssd-pretrain/seglink

# get the number of gpus
OLD_IFS="$IFS"
IFS=","
gpus=($CUDA_VISIBLE_DEVICES)
IFS="$OLD_IFS"
NUM_GPUS=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $NUM_GPUS \* $IMG_PER_GPU`

#dataset
if [ $DATASET == 'synthtext' ]
then
    DATA_PATH=SynthText
elif [ $DATASET == 'scut' ]
then
    DATA_PATH=SCUT
elif [ $DATASET == 'icdar2013' ]
then
    DATA_PATH=ICDAR
elif [ $DATASET == 'icdar2015' ]
then
    DATA_PATH=ICDAR
else
    echo invalid dataset: $DATASET
    exit
fi

DATASET_DIR=$HOME/dataset/SSD-tf/${DATA_PATH}

python train_seglink.py \
			--train_dir=${TRAIN_DIR} \ # 存放checkpoint和summery
			--num_gpus=${NUM_GPUS} \ # GPU个数
			--learning_rate=0.0001 \
			--gpu_memory_fraction=-1 \
			--train_image_width=384 \
			--train_image_height=384 \
			--batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \ #数据路径
            --dataset_name=${DATASET} \ # 数据scope名字
            --dataset_split_name=train \
            --train_with_ignored=0 \
			--checkpoint_path=${CHKPT_PATH} \ # 模型文件存储位置
			--using_moving_average=0 # 是否使用滑动平均
