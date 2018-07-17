source activate mxnet-p36
pip install mxnet-cu90
pip install gluoncv

cd ~
git clone https://github.com/dmlc/gluon-cv
git clone https://github.com/duducheng/deeplabv3p_gluon
mkdir weights

cd gluon-cv/scripts/datasets
python pascal_voc.py

cd ~/weights

function gdrive_download () {
 CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
 wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
 rm -rf /tmp/cookies.txt
}

gdrive_download 19zxsJ6tmPuJcEBd-P93yCEFMLc7o4dPP pascal_train_aug.params
gdrive_download 15kZ90a0Fnz4j7XryHb41gNVzFtecHqLR pascal_trainval.params
gdrive_download 1XAtN7My93Zk4p23kryp4ytFvvQKwU7RO imagenet_pretrain_for_voc.params


