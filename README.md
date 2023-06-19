# Classifier_Pytorch

This code is an implementation of a deep learning model using transfer learning with ResNet50 architecture to classify images into two categories. The categories are defined by the subfolders within three directories, train, validation, and test. The model is trained on the images in the train directory and validated on the images in the validation directory. The testing dataset is used to test the model's performance after training and validation.

The code begins by loading the required libraries, including torch, torchvision, numpy, pandas, etc. Then it defines transformations for the training, validation, and testing data. It uses the datasets.ImageFolder class to create datasets from the images in the directories.

After that, the code creates dataloaders for training, validation, and testing data. Then it loads the pretrained ResNet50 model using the models.resnet50 function, freezes the model's parameters, and modifies the output layer to classify the images into two categories.

The code then defines the loss function and optimizer, as well as two functions, Sigmoid and validation, used for validation. Sigmoid applies the sigmoid function to the model's output and converts the probabilities into a binary output. validation function computes the validation loss, accuracy, and predictions on the validation dataset.

Finally, the code runs the training and validation loop for 100 epochs, with each epoch consisting of a forward pass, backward pass, and optimizer step. The training loss, training accuracy, validation loss, and validation accuracy are recorded for each epoch. The performance of the model is evaluated on the test dataset after the training loop has finished.
