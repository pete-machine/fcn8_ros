# Semantic Segmentation in ROS
Implementing a ros package for interfacing FCN-8 with ROS. FCN-8 is described in ([link](https://arxiv.org/abs/1605.06211)) and the original code (not made by me) is available at ([link](https://github.com/shelhamer/fcn.berkeleyvision.org)).

(Tested on Ubuntu 16.04, ROS Kinetic, Python 2.7.12, opencv 3.2.0-dev, Cuda 8.0, cuDNN 6.0)

## Install dependencies
#### ROS kinetic
Follow the ROS [installation guide](http://wiki.ros.org/ROS/Installation) for download and installation.

#### Get nvidia drivers (Only if GPU is used)

	sudo add-apt-repository ppa:graphics-drivers/ppa
	sudo apt-get update

Get the newest driver with the “Software & Updates” application in the “Additional Drivers”-tab.

#### CUDA 8.0 (Only if GPU is used)
Software requires CUDA to be installed. 
Go to [download page](https://developer.nvidia.com/cuda-downloads) select your platform, download and follow instructions.
In Select Target Platform i used Linux/x86_64/16.04/deb (local)

#### cuDNN 6.0 (Only if GPU is used)
(You will need a nvidia user) 
Download cuDNN [link](https://developer.nvidia.com/rdp/cudnn-download).
Extract zipped folder and copy to /usr/local/cuda-8.0

	cd ~/Downloads/cuda (This is the directory of the extracted folder)
	sudo cp ./include/* /usr/local/cuda-8.0/include/
	sudo cp ./lib64/* /usr/local/cuda-8.0/lib64/

#### Caffe
Install caffe dependencies

	sudo apt-get install libboost-dev libprotobuf-dev protobuf-compiler libgflags-dev libgoogle-glog-dev libblas-dev libhdf5-serial-dev libopencv-dev libleveldb-dev liblmdb-dev libboost-all-dev libatlas-base-dev libsnappy-dev python-pip
	pip install scipy
	pip install numpy
	pip install -U scikit-image
	pip install protobuf
	//apt-get install python-opencv


#### Caffe 
Get nvidias edition of caffe and git clone it to ~/Code/

	cd && mkdir Code && cd Code 

You may either select original caffe version from Berkeley (__Caffe (Berkeley)__) or nvidias edition (__Caffe (nvidia)__).
	git clone https://github.com/BVLC/caffe.git # Caffe (Berkeley)
	git clone https://github.com/NVIDIA/caffe.git  # Caffe (nvidia)

In Makefile replace LIBRARIES line (around 183) with 

	LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial

Make a copy of Makefile.config.example
	
	cd ~/Code/caffe
	cp Makefile.config.example Makefile.config

Add /usr/include/hdf5/serial/ to INCLUDE_DIRS (around line 94-96)

	INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/


#### Same for both Caffe (nvidia) and Caffe (berkeley)

Include the following lines in the Makefile.config 

	USE_CUDNN := 1 	(Only if GPU is used)
	WITH_PYTHON_LAYER := 1

RESTART computer before building caffe
	sudo reboot now

Build caffe

	cd ~/Code/caffe
	make all -j8
	make test -j8
	make runtest -j8
	make pycaffe

	One-liner
	make all -j8 && make test -j8 && make runtest -j8 && make pycaffe

Append the following lines to ~/.bashrc

	# CAFFE 
	export PATH=/usr/local/cuda/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
	export CAFFE_ROOT=~/Code/caffe # (IMPORTANT TO SET THIS AS THE INSTALL DIRECTORY OF CAFFE)
	export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH

## Install semantic segmentation
Git clone the package to your ros-workspace repository. 

	cd <your_workspace>/src
	git clone https://github.com/PeteHeine/RosSemanticSegmentation

Get weights. (Prototxt and caffemodel is from [here](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/pascalcontext-fcn8s))
	
	cd <your_workspace>/src/fcn8_ros/
	wget http://dl.caffe.berkeleyvision.org/pascalcontext-fcn8s-heavy.caffemodel




