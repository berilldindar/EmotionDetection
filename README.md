# EmotionDetection
Dataset Link:https://www.kaggle.com/msambare/fer2013


Deep Learning techniques have started to take place in many areas of our lives. It
One of the fields is Emotion Recognition Systems. Emotion recognition systems, given by an emotion
big data labeled with these emotional states
learns to determine with neural networks trained from sets. Emotion Recognition systems
There are many areas where it is used. These; Security, Recruitment, Customer
Satisfaction, Socialization of children who need special attention.
Our emotions determine our perspective on events. We experience the most emotional change.
One of the areas is business life, which covers the majority of our lives. positive in business
We may experience negative reactions as well as negative events and in such cases stress,
We should be able to control our sadness and anger. Human Resources while recruiting
department for that position not only our technical competence, but also our competence in many areas.
is measuring. One of these areas is Emotion Recognition, that is, Personality, as we are called.
Inventory Tests. It is a test that measures whether we are also emotionally ready for that position.
Some Companies have Deep Learning models capable of emotion recognition as HR assistants
they are using. This artificial intelligence system is in the first and most time-consuming phase of the selection process.
is used. Keywords, intonation, and facial expressions of applicants
is evaluating.

The data is prepared for training by passing through the CRISP-DM stages. CRISP-DM 6 stages
occurs. These;
Business Understanding
It is the most important stage. Understanding the purpose and requirements of the project from a business perspective
• Defining business goals and success criteria
• Conducting the situation assessment
• Determining the aims of the project
• Creation of the project plan
Data Preparation
• Extracting data from the data warehouse
• Consolidation of data files from different systems
• Inconsistent variable values ​​become consistent
• Identification of missing, incorrectly entered or outliers
• Data selection
• Conversion of related variables
Modeling
Analysis methods are used to extract the necessary information from the data. This stage model
selection of techniques, production of test design, creation and evaluation of the model
Evaluation
Before applying the model, the model should be evaluated in detail and the model created
is the re-examination phase.
Application (Deployment)
If the purpose of a model is to increase data knowledge, the knowledge gained must be organized and decision-making.
It should provide a way to use the organization in giving.
Image processing, Deep Learning for Computer Vision projects
One of the techniques of Convolutional Neural Network (Convolutional Neural Networks) algorithm
is used.

STRUCTURE OF Convolutional NEURAL NETWORKS
CNN processes the image with various neural network layers. Basically classification
Neural networks are used to solve the problem.
Convolutional Layer
Convolutional layers are the main building block of CNN. properties of the given picture.
It is the layer responsible for sensing. In this layer, the low and high levels of the image
Applies some filters to the image to extract features. This will detect edges for images
It could be a filter. These filters are usually multidimensional and contain pixel values.
The height, width and depth of the created matrices are represented.
When we apply it, we get a Feature map output created by the features. Each filter
Feature map is updated when applied. How does stride work around the filter's input image?
checks for evolution. Stride size determines the size of the Feature Map.
Non-Linearity
Usually non-linearity after all convolutional layers
layers are written. These layers each use an activation function, optimizing the model.
aims to reduce costs. The activation function is when a neuron fires and
determines the value range of the output. Usually the threshold value (bias) to the activation function
The final value of the activation function is obtained by adding another term called
Along with its activation functions, many different types such as tanh, sigmoid, and recently relu
optimization function is used.
ReLu Function f (x) = max (0, x) Mathematical representation is made.
Pooling Layer
It helps to reduce the size of the volume of the input layer. Weight of parameters
It helps to reduce the amount of To prevent overfitting
is a method used.
The task of this layer is to determine the displacement size of the representation and the parameters within the network and the computation
to reduce the number. In this way, incompatibility in the network is checked. Many Pooling
There are transactions, but the most popular is Max Pooling. Average working on the same principle
There are also Pooling, and L2-norm Pooling algorithms.
As a disadvantage, it can degrade the quality of the image.
Flattening Layer
The task of this layer is the Fully Connected Layer, which is the most important layer of the neural network model.
Prepares the data at the (Full Dependent Layer) input. Neural networks combine input data into a one-dimensional
it's like a string. In CNN, the matrices coming from the Convolutional and Pooling layer are unique.
It converts it into a one-dimensional array.
Fully Connected Layer
This layer receives the data after Flattening. neural network with dense layers
performs the training.


I wanted to briefly explain the general structure of CNN networks. Will examine my project in detail
If we are;
I realized my project in Colaboratory environment. With free GPU support
It enabled me to easily print out the data much faster than my means.
Fer2013 dataset is a widely used dataset in sentiment analysis projects.
I learned this and started my project by uploading this dataset from Kaggle to my Drive. Data,
consists of 48x48 pixel grayscale images of faces. faces, up and down
automatically centered and occupying approximately the same amount of space on each image.
was recorded as. The task ate each face according to the emotion shown in the facial expression.
category (0 = Angry, 1 = Disgusted, 2 = Fear, 3 = Happy, 4 = Sad, 5 =
Surprised, 6 = Natural). The training set consists of 28,709 samples and the public test set consists of 3,589 samples.
consists of.
I imported the libraries that I will use during the project. I then read the dataset
My dataset consists of 35887 rows and 3 columns. What are the pixel values ​​of the image and what emotion?
I shared a small area from the dataset to show that it is tagged with .
I parsed the dataset to be used for training and testing and counted their numbers with Usage.
I explained. With Usage, you can determine how many groups the train and test numbers are divided into in the data set.
we can observe. Private and public test so we don't use it in datasets in Kaggle
We observed the numbers in the data set in the section separated as
Training dataset: 28709
Since the pixel values ​​of the samples in the training set are in tabular form, you can parse them with a split
I assigned it to the list variable. I got the pixel values ​​of all the images.
I reshape the images to get a 48x48 image. From the images in the dataset
I took the screenshot to be able to check any of them. Number of different moods
I printed it out and that way I kept checking the contents of the dataset.
I determined the mood for each sample in the Train data and did the same for the test dataset.
I also did. I separated the Public and Private tests and completed the preprocessing steps.
After determining the shapes for the train and test data, I decided to create a model.
Convolution Neural as the most suitable algorithm for computer vision projects will be CNN
I created a model using Network. Since there are 7 emotion definitions, I created 7 layers.
'relu' for Convolution, BatchNormalization and activation function in layers
i used it. In the 2nd layer, I also applied MaxPooling and Dropout operations and
I used the activation function 'relu' for the layer. (Except the last layer)
In the last layer, the loss function is called as Multi Class Single Label.
Since we will label it, I set the loss function as 'categorical_crossEntropy'.
I used softmax as the classifier, I used man as the optimizer, and I used the success metric.
I set it to accuracy. Since the total output category will be 7, we will set our Dense number as 7
I set it. I shared the summary of the model.
After creating the model and before training the model, I again have my train and test data
I did the check. Apart from the CNN model, it can also be modeled with many architectures (VGG, ResNet50).
I've tried. I observed that the VGG architecture is not at all suitable for this dataset. CNN and ResNet
Although it has almost the same success rate, it has the most successful prediction in the test results.
The architecture that does it is the ResNet architecture. I trained my model for 10 epochs and mixed
I got him to get the images. The accuracy and validation accuracy values ​​I obtained are combined
I showed it graphically.
Finally I added many different images to test the model I trained and
I tested the image. It predicts mood with a large percentage in most photos.
performed successfully.

