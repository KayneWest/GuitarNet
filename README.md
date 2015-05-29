# GuitarNet

VGG style net to classify Guitars and their worth

Need to have CUDA installed and a GPU machine.
I used an AWS Ubuntu GPU machine. 
Here are some good resources for CUDA:
    
    http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.0-ubuntu
    http://askubuntu.com/questions/451672/installing-and-testing-cuda-in-ubuntu-14-04
    https://groups.google.com/forum/#!msg/theano-users/xW9jmHzOwp0/8SvMA_R0EAUJ
    https://github.com/Theano/libgpuarray/issues/19

To run this on a basic AWS GPU instance:
    
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nodatanet.py
