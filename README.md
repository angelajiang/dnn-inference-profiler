# TensorFlow

## Installing Tensorflow
[Instructions](https://www.tensorflow.org/install/)

## Image classification simple
```bash
cd tensorflow
python classify_images.py --image_dir /path/to/image/dir
```

## Image classification using TF Slim
### Clone TensorFlow Slim
```bash
git clone https://github.com/angelajiang/models.git
cd models/slim/
```

### Download pretrained checkpoint files
[Details](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)
```bash
mkdir /tmp/checkpoints/
export CHECKPOINT_DIR=/tmp/checkpoints
```

#### Download InceptionV3
``` bash
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvf inception_v3_2016_08_28.tar.gz
mv inception_v3.ckpt ${CHECKPOINT_DIR}
rm inception_v3_2016_08_28.tar.gz
```

#### Download ResNetv2
``` bash
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
tar -xvf inception_resnet_v2_2016_08_30.tar.gz
mv inception_resnet_v2_2016_08_30.ckpt ${CHECKPOINT_DIR}
rm inception_resnet_v2_2016_08_30.tar.gz
```

### Prepare datasets ([details](https://github.com/tensorflow/models/tree/master/slim#preparing-the-datasets))
``` bash
export DATA_DIR=/path/to/data/dir
python download_and_convert_data.py --dataset_name=flowers --dataset_dir="${DATA_DIR}"
python create_validation_split.py $DATA_DIR
```

### Perform evaluation using resnet
``` bash
python eval_image_classifier.py --checkpoint_path="${CHECKPOINT_DIR}" \
                                --dataset_dir=/path/to/image-data/  \
                                --dataset_name=flowers \
                                --dataset_split_name=train \
                                --model_name=inception_resnet_v2 \
                                --batch_size=2  # ResNet
# Toggle GPU off to use CPU
export CUDA_VISIBLE_DEVICES=""  # Make GPU invisible
# run code
unset CUDA_VISIBLE_DEVICES      # Return to normal
```
