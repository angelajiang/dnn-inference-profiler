# TensorFlow

## Image classification simple
```bash
cd tensorflow
python classify_images.py --image_dir /path/to/image/dir
```

## Image classification using TF Slim
```bash
# Clone TensorFlow Slim
git clone https://github.com/angelajiang/models.git
cd models/slim/

# Follow [instructions](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)
to download pretrained .ckpt files into $CHECKPOINT_DIR

# Prepare datasets ([details](https://github.com/tensorflow/models/tree/master/slim#preparing-the-datasets))
export DATA_DIR=/path/to/data/dir
python download_and_convert_data.py --dataset_name=flowers --dataset_dir="${DATA_DIR}"
python create_validation_split.py $DATA_DIR

# Perform evaluation using resnet
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
