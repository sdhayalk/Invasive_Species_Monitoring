# Invasive Species Monitoring

Implementation of GoogLeNet architecture with Inception modules and Ensemble of Inception V3, Xception, ResNet50 using transfer learning to detect invasive hydrangea in Brazilian national forest images dataset from Kaggle.

Programmed in Keras using TensorFlow backend.

Made this project for participating in Kaggle competition Invasive Species Monitoring. Find details about the competition [here](https://www.kaggle.com/c/invasive-species-monitoring)

### Prerequisites
Python, TensorFlow, Keras (and other libraries such as NumPy, etc.). If using GPU for training, you will need an NVIDIA GPU card with software packages such as CUDA Toolkit 8.0 and cuDNN v5.1. See [here](https://www.tensorflow.org/install/install_linux) for more details.

Download the dataset from [here](https://www.kaggle.com/c/invasive-species-monitoring/data).

## Input
Input are various 1154x866 resolution RGB images (three channels), along with label corresponding to whether the image has presence of the invasive species hydrangea

## Results
Achieved 0.977 and 0.987 area under ROC curve with GoogleNet and the ensemble respectively. Including Batch Normalization and Dropout layers after some Convolution layers in GoogleNet drastically increased the accuracy and overcame the problem of only one class being predicted for all examples.

### References and Acknowledgments
1] Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Zhou, Y. (2013, November). Challenges in representation learning: A report on three machine learning contests. In International Conference on Neural Information Processing (pp. 117-124). Springer, Berlin, Heidelberg.

2] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

3] Pramerdorfer, C., & Kampel, M. (2016). Facial Expression Recognition using Convolutional Neural Networks: State of the Art. arXiv preprint arXiv:1612.02903.
