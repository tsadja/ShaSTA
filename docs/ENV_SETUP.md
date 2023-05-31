# Environment Setup
First, clone the nuScenes dev-kit.
```bash
git clone https://github.com/nutonomy/nuscenes-devkit.git
```

If you do not have CUDA 11.0 installed, you can easily install it by following [this link](https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal).

Add the following lines to your .bashrc file. We use CUDA 11.0, but if you use a different version please make adjustments accordingly. Make sure to replace /PATH-TO with your home directory.
```bash
export PYTHONPATH="${PYTHONPATH}:/PATH-TO/nuscenes-devkit/python-sdk"
export PATH=/usr/local/cuda-11.0/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.0
export CUDA_HOME=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-7
export CXX=/usr/bin/g++-7
```

We use conda environments, so please make sure to install conda and update your .bashrc accordingly. [This link](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) contains the instructions for installing conda. Note that we find Miniconda sufficient. 

Set up the conda environment with the following lines:
```bash
conda env create -f environment.yml
pip install -r requirements.txt
pip install torch==1.7.0 torchvision -f https://download.pytorch.org/whl/cu110/torch_stable.html
git clone https://github.com/NVIDIA/apex
cd apex # this should be located in ~/ShaSTA/apex
git checkout 5633f6  # recent commit doesn't build in our system
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../det3d/ops/iou3d_nms
python setup.py build_ext --inplace
```

