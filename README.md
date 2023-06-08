# Segmentation Code for B@I1 Group in UW.

Jinpeng would like to thank the friendly and kind BAIL group members at UW. 
Here is a TensorFlow-based Deep Learning code for Retina Layer Segmentation.
The ReadMe will contain two sections: 1) Environment Setup; 2) How to Use this Code. 
If you want to use the below code for OCT retinal layer segmentation, please follow the instruction.

# Environment Setup
0. Suggest Hardware: 
- Intel Core-series CPU
- NVIDIA (Geforce) Graphics Card

1. Installation of Software:
- Installation of Anaconda: Please install Anaconda from offical website: https://www.anaconda.com/
- Installation of PyCharm (hint: you can use uw.edu email to request a linscen of professional version) 
URL: https://www.jetbrains.com/pycharm/download/#section=windows

2. Setup your environment based on Anaconda:
- Open your Anaconda3, and select the 'Environments' in the left hand bar.
- In the 'Environments' Page, you should then click the 'Create' botton.
- You will see a window with (Name, Location, and Packages). Named your virtual environment, and select Python 3.9.XX Version. Finally, please click 'Create' and wait for Anaconda. 
- After the virtual environment set, you should open the command window by click the 'play button'-like botton. 

3. Install CUDA-Toolkit and CUDNN, and TensorFlow 2.9.0.  
(Introduction: CUDA-Toolkit and CUDNN are packages provided by NVIDIA that can help you run the deep-learning code based on GPU) 
Now we assume that you are in the command window of your vitural environment. Please copy below command to install package.
- conda install cudnn
- pip install tensorflow-gpu==2.9.0
- pip install tensorflow_addons==0.18.0
- pip install opencv-python
- pip install scikit-learn
- pip install scipy
- pip install matplotlib
- pip install tqdm

Please mannually install additional packages if there is a error raise by Python. 

4. Please read detailly to understand how to configure a conda virtual environment.
- Find 'Existing conda environment' in the following URL and configure your created virtual environment in Step 2&3 above: https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html 

# Run the Code
To easily run the code for segmentation task demo, please follow the below instruction step-by-step.

(1) Find the 'Config.py' in folder 'Configuration', then change the parameter which marked with 'TODO'.

(2) Open the 'Main.py' and change the filepath of image data and label data; then run the 'Main.py'.

hint: a demo-used data is available in the folder 'Dataset->retina seg test data'.
hint: further details will be available in the comments of the code. 

