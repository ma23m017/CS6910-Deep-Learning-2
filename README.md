# CS6910-Deep-Learning-2

## Part A

In this part I am training a CNN model from scratch and tuning the hyperparameters so that the model performs well. I have used python for my implementations. I have used the required packages from PyTorch and Torchvision.

### Working:
I have made a CNN model which consists of 5 convolutional layers, after each convolution layer I have added an activation and a max-pooling layer which together makes a block. After 5 such blocks there is 1 dense layer and then finally 1 output layer with 10 neurons, 1 for each classes.
Then I have trained my model using the iNaturalist dataset, where I have splitted the training data in 80:20 for training the data and validating the data. I also tried to find the best hyperparameter configuration by using sweep feature provided by wandb. I have taken some hyperparameters which can be seen below:
### Hyperparameters and their values:
1. Kernel size(size of filters): [[3,3,3,3,3], [3,5,5,7,7], [7,7,5,5,3]]
2. Drop out : [0.2, 0.3]
3. Activation function: ['ReLU', 'GELU', 'SiLU', 'Mish']
4. Batch normalization: [True, False]
5. Filter organization: [[32,32,32,32,32], [128, 128, 64, 64,32], [32, 64,128,256,512]]
6. Data augmentation: [False]
7. Number of neurons in dense layer: [128, 256]

After choosing the hyperparameters value for which I got the best validation accuracy and I consider it as ny best model. Then I applied my best model on the test data and I also has reported the accuracy on the test data.

### Best hyperparameters' value:
1. Kernel size(size of filters):
2. Drop out :
3.  Activation function:
4. Batch normalization:
5. Filter organization:
6. Data augmentation:
7. Number of neurons in dense layer:


## Part B

In this part we have been asked to use a pre-trained model instead of training the model from the scratch. So, I have loaded a model(resnet50) from torchvision which is pre-trained on the ImageNet dataset which is somewhat similar to iNaturalist data set.

### Working:
I have loaded a pre trained model i.e. ResNet50 and then fine tuned it using the same data i.e naturalist data. I have used the eights which I got from from training the model on ImageNet data. I also have done the resizing of the image to get the dimension of image in ImageNet data equal to image in my data. I also have made the number of neurons in output layer equal to 10 which was 1000 before to match the number of classes in naturalist dataset which has only 10 classes. Then in fine tuning part I have used different strategies and fine-tuned the model using the iNaturalist dataset, that strategies are given as:
1. Freezing all layers except the last layer
2. Freezing upto k layers and fine tuning the rest
3. Freezing the rest

After that I have done the process by freezing 70, 80 and 90 percent of the layers and also printed the validation acccuracies for different run count.

### References:




