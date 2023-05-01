# Plant_Disease_Classification
# Introduction

Plant growth can be affected by many types of diseases which directly leads to effect the crop production and that affect the supply chain in food industry. We have a lot of option to decrease the plant dieseas but we need to take that option which is most efficient in both time and money.

With the help of deep learning technologies we can reduced the time of searching which plant's leaves are diseased or healthy. It's a typical classification problem that can be solved using many of the deep learning algorithm such as DNN, CNN, or other machine learning algorithms.

In this project we are using CNN to classify which plant leaves are healthy and which one is not. Moreover we are using state of art transfer learning techniques such as VGG16 to get highly accurate model that precisiely classify the image.

# Understanding CNN

It stands for Convolution Neural Network, and it is the best algorithm when it comes to working with images, basically it takes two major mathematical opration that diffrentiate it with other Neural Network techniques.

    Convolution Opration
    Pooling Opration

# 1. Convolution Opration:
       
Convolution is a specialized kind of linear operation. Convolution between two functions in mathematics produces a third function expressing how the shape of one function is modified by other.
        
# Convolution Kernels
A kernel is a small 2D matrix whose contents are based upon the operations to be performed. A kernel maps on the input image by simple matrix multiplication and addition, the output obtained is of lower dimensions and therefore easier to work with.

we found that our input matrix is of 6x6 and filter is of size 3x3 with stride = 1 and padding = 0, * represents convolution operation between Input matrix and the filter. This filter is basically used to detect the vertical edge in the image i.e. resultant matrix is basically used to reduced the image width and only take those part which is important.

# 2. Pooling Operation

Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network. We uses pooling to recognize an image, if the image is tilted or not same as previous image.

There are basically 2 types of pooling opraration:

    Max Pooling
    Average Pooling

# 1. Max Pooling:

this indicates if a 2x2 Max pool is used in the network then the matrix creates a 2x2 window and takes the maximum value amoung the 4 values in that particular window. It's very important opration in CNN because it's bascally removes those low parameter values and reduces computation.

# 2. Average pooling: 

Average pooling is also doing similar opration but instead of taking maximum value from the window it calculates the average of the window and then gives the result. Basically today's in general we are using max pooling as the pooling layer opration because it gives better accuracy and also it's little faster than the average pooling opratin

With this two operation in CNN we can able to compute 2D inputs such as images very easily.
# Complete CNN architecture
Let me explain the steps involved in this architecture

   In first step an images is passed to Conv layer 1 which is used to do convolutin operation
   Then pooling layer is created to reduced parameters
   Layer 3 and 4 are similar like 1 and 2
   In layer 5 which termed as hidden in this image also called flatten on fully connected layer are just a dense layer converted from the last conv layer after this layer only we apply sigmoid or softmax activation funtion to get the output.

Let's now talk about VGG16 
# Understanding VGG16 architecture

VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

# What is Transfer Learning?

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems. We are using the same approach for training our model. For this task we are basically following VGG16 architecure with pretrained model with imagenet.

# Conclusion
 
 The proposed work aims to develop a plant disease classification system using the VGG-16 architecture, which is a deep learning model known for its ability to extract and classify complex features in images. The system takes an input image of a diseased plant, preprocesses it, and extracts features using the VGG-16 model. The extracted features are then passed to a fully connected classification layer, which outputs the predicted class label for the input image.One potential area for future research is to explore other deep learning architectures, such as ResNet and Inception, and compare their performance with the VGG-16 architecture in the context of plant disease classification. Another direction for future work could be to collect a larger and more diverse dataset of plant disease images to improve the accuracy and robustness of the classification system. Additionally, the development of a user-friendly interface for the system could increase its accessibility and usefulness for farmers and agricultural researchers. Finally, the system's performance could be evaluated under various environmental and lighting conditions to determine its feasibility and reliability in real-world settings.
