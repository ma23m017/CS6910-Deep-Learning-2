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
6. Number of neurons in dense layer: [128, 256]

After choosing the hyperparameters value for which I got the best validation accuracy and I consider it as ny best model. Then I applied my best model on the test data and I also has reported the accuracy on the test data.

### Best hyperparameters' value:
1. Kernel size(size of filters):
2. Drop out :
3.  Activation function:
4. Batch normalization:
5. Filter organization:
6. Number of neurons in dense layer:

Note: I have not not included data augmentation as hyperparameter.

### Refrences:


## Part B

In this part we have been asked to use a pre-trained model instead of training the model from the scratch. So, I have loaded a model(resnet50) from torchvision which is pre-trained on the ImageNet dataset which is somewhat similar to iNaturalist data set.
