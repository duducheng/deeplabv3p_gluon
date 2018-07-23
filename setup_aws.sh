source activate mxnet_p36

git clone --recursive https://github.com/apache/incubator-mxnet
cd incubator-mxnet
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
cd python
pip install -e .

pip install gluoncv

cd ~
mkdir weights
git clone https://github.com/dmlc/gluon-cv
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

git clone https://github.com/duducheng/deeplabv3p_gluon

pip uninstall -y mxnet-cu90mkl
python -c "import mxnet; print(mxnet.__version__)"
