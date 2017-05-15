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

# Caffe

## Installing Kernel Patch tips
```bash
> kernel/bounds.c:1:0: error: code model kernel does not support PIC mode
sudo apt install gcc-4.9
vim Makefile
r/gcc/gcc-4.9
```

## OpenCL
### Find OpenCL when building Caffe
``` bash
vim cmake/Modules/FindOpenCL.cmake
```
```
FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include" "/opt/intel/opencl/include/")
FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include" "/opt/intel/opencl/include/")
```
``` bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/opencl/
```

## ViennaCL
```bash
cd
git clone https://github.com/viennacl/viennacl-dev/tree/release-1.7.1
cd viennacl-dev
sudo cp -r viennacl /usr/local/include
```

## BVLC CAFFE Python Interface
``` bash
make pycaffe
export PYTHONPATH=/path/to/caffe/python/
pip install -U scikit-image
pip install tensorflow
```

## Running inference using pretrained models

```bash
vim caffe/python/caffe/io.py    #Fix bug in io.py
if ms != self.inputs[in_][1:]:
		print(self.inputs[in_])
		in_shape = self.inputs[in_][1:]
		m_min, m_max = mean.min(), mean.max()
		normal_mean = (mean - m_min) / (m_max - m_min)
		mean = resize_image(normal_mean.transpose((1,2,0)),in_shape[1:]).transpose((2,0,1))
		#raise ValueError('Mean shape incompatible with input shape.')                      # Original line

vim caffe/python/caffe/classifier.py    #Only time forward pass
import time

start = time.time()
out = self.forward_all(**{self.inputs[0]: caffe_in})																		# Original line
print("Core in %.2f s." % (time.time() - start))

```

### GoogLeNet
```bash
cd caffe/python
python classify.py ~/image_data/architecture-benchmark/ output_file --pretrained_model ../models/bvlc_googlenet/bvlc_googlenet.caffemodel \
                                                                    --model_def="../models/bvlc_googlenet/deploy.prototxt" \
                                                                    --mean_file="caffe/imagenet/ilsvrc_2012_mean.npy"
```

### ResNet
[Links](https://github.com/KaimingHe/deep-residual-networks) to .caffemodel files
```bash
git clone https://github.com/KaimingHe/deep-residual-networks.git
cd deep-residual-networks
cp prototxt/ResNet-50-deploy.prototxt /path/to/models/
```
```bash
cd caffe/python
python classify.py ~/image-data/architecture-benchmark/ output_file --pretrained_model ../models/ResNet/ResNet-50-model.caffemodel \
                                                                    --model_def="../models/ResNet/ResNet-50-deploy.prototxt" \
                                                                    --mean_file="caffe/imagenet/ilsvrc_2012_mean.npy"
python classify.py ~/image_data/architecture-benchmark/ output_file --pretrained_model ../models/ResNet/ResNet-50-model.caffemodel \
                                                                    --model_def="../models/ResNet/ResNet-50-deploy.prototxt" \
                                                                    --mean_file="caffe/imagenet/ilsvrc_2012_mean.npy"

python classify.py ~/image_data/architecture-benchmark/ output_file --pretrained_model ../models/ResNet/ResNet-50-model.caffemodel \
                                                                    --model_def="../models/mkl2017_resnet_50/solver.prototxt" \
                                                                    --mean_file="caffe/imagenet/ilsvrc_2012_mean.npy"
```

## Installing bvlc caffe on Orca

```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libopenblas-dev
sudo apt-get install libatlas-base-dev
```
