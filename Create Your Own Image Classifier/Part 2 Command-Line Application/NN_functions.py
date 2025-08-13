#Importing libraries

#For Neural Network (NN) Model and Data Handling
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

#For Image Classification Models
from torchvision import datasets, transforms, models

#External Libraries Needed
from collections import OrderedDict
import json
import argparse
import os

#For Image Processing
from PIL import Image
import matplotlib.pyplot as plt

#For Array Calculations
import numpy as np

def device():
    '''
    Function: Identifies a GPU and uses it if available.

    Inputs: None

    Return: device variable with appropriate GPU string
    '''
    
    #For Mac with MPS GPU
    mps = torch.mps.is_available()
    
    #For CUDA GPU
    cuda = torch.cuda.is_available()
    
    if mps:
        device = torch.device("mps")
    
    elif cuda:
        device = torch.device("cuda")
    
    else:
        device = "cpu"
    
    return device

def get_input_train():
    '''
    Function: Takes in command-line arguments to train NN model.

    Inputs:
    data_dir - str of directory where data is located
        
        Optional:
        --save_dir - str of directory name where model checkpoints should be stored
        --arch - str of NN architecture selected for image classification
        --learning_rate - int to set learning rate of the model
        --hidden_units - int of number of hidden units in the hidden layer
        --epochs - int of number of training cycles
        --gpu - bool wheather to use GPU for training


    Return: parse.parse_args() for command-line parsing
    '''

    #Set Up a Parser
    parser = argparse.ArgumentParser(description='Command Line Application of Image Classification NN Model for Udacity Course')

    #Get Current Directory
    current_dir = os.getcwd()

    #Check if data folder exists to run script
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)
    
    def minimum_lr(arg):
        """Type Function to ensure int is within expected bounds. Ensure learning rate is greater than 1."""
        try:
            value = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if value < 0.000:
            raise argparse.ArgumentTypeError("Learning rate must be greater than 0.000.")

        return value

    def minimum_hu(arg):
        """Type Function to ensure int is within expected bounds. Ensure there is at least 1 hidden unit in the hidden layer."""
        try:
            value = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a int number")
        if value < 0:
            raise argparse.ArgumentTypeError("Number of Hidden Units must be greater than 1.")

        return value
    
    def minimum_e(arg):
        """Type Function to ensure int is within expected bounds. Ensure there is at least 10 training cycles."""
        try:
            value = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a int number")
        if value < 0:
            raise argparse.ArgumentTypeError("Number of Hidden Units must be greater than 1.")

        return value

    #Command Line String Arguments
    parser.add_argument("data_dir", type=dir_path)

    parser.add_argument("--save_dir", type=str, default=current_dir, help="Set Directory to Save Checkpoints. Default is current working directory")
    parser.add_argument("--arch", type=str, default="resnet101", help="Choose NN Architecture for Image Classification (VGG11 or Resnet101)")


    #Command Line Int Arguments with checks for minimum values
    parser.add_argument("--learning_rate", type=minimum_lr, default=0.002, help="Set learning rate for NN Model. Default is 0.002.")
    parser.add_argument("--hidden_units", type=minimum_hu, default=512, help="Set number of hidden units in the hidden layer for NN Model. Default is 512 units.")
    parser.add_argument("--epochs", type=minimum_e, default=20, help="Set number of training cycles for NN Model. Default is 20 epochs. Minimum is 10 epochs.")
    
    #Command Line Bool Arguments
    parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU for training.")

    return parser.parse_args()

def get_input_predict():
    '''
    Function: Takes in command-line arguments to predict image name using trained NN Model.

    Inputs:
    data_dir - str of directory where data is located
        
        Optional:
        --save_dir - str of directory name where model checkpoints should be stored
        --arch - str of NN architecture selected for image classification
        --learning_rate - int to set learning rate of the model
        --hidden_units - int of number of hidden units in the hidden layer
        --epochs - int of number of training cycles
        --gpu - bool wheather to use GPU for training


    Return: parse.parse_args() for command-line parsing
    '''

    #Set Up a Parser
    parser = argparse.ArgumentParser(description='Command Line Application of Image Classification NN Model for Udacity Course')

    #Get Current Directory
    current_dir = os.getcwd()

    #Command Line String Arguments
    parser.add_argument('path_to_image')
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Set mapping categories to real names.")


    def minimum_k(arg):
        """Type Function to ensure int is within expected bounds. Ensure there is at least 10 training cycles."""
        try:
            value = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a int number")
        if value < 1:
            raise argparse.ArgumentTypeError("Number of K classes predicted must be greater than or equal to 1.")

        return value

    #Command Line Int Arguments
    parser.add_argument("--top_k", type=minimum_k, default=5, help="Return top K most likely classes for image provided. K must be at least 1.")

    #Command Line Bool Arguments
    parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU for training.")

    return parser.parse_args()

def load_data(data_dir):
	'''
	Function: Takes str input for data image folder and ensures data is split accordingly for NN usage.
	Performs data transformations for the training and validation datasets accordingly. Lastly, it takes
	the training dataset and validation dataset and creates DataLoaders for each dataset respecively.

	Inputs: str with location of data folder

	Return: training DataLoader, validation DataLoader, test DataLoader, class mapping
	'''

	data_dir = data_dir
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	#Perform random transformations on training datset, resize images to 224 x 224 pixels for model image classification.
	train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

	#Resize images (224 x 224) for validation dataset
	val_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

	#Load datasets using ImageFolder class
	train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
	val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)
	test_data = datasets.ImageFolder(test_dir, transform=val_transforms)

	#Using the image datasets, create the DataLoaders
	trainloader = DataLoader(train_data, batch_size=40, shuffle=True)
	valloader = DataLoader(val_data, batch_size=40)
	testloader = DataLoader(test_data, batch_size=40)

	#Index Class labels for easier classification
	class_mapping = test_data.class_to_idx


	return trainloader, valloader, testloader, class_mapping

def load_categories(label_names):
    '''
	Function: Takes .json file with class labels and creates a dict holding the class labels for classifcation.
	Retrive number of classes used for the dataset.

	Inputs: .json file label_names

	Return: cat_to_name dictionary, int num_classes
	'''
    
    with open(label_names, 'r') as f:
        cat_to_name = json.load(f)
        
    num_classes = len(cat_to_name.keys())

    return cat_to_name, num_classes

def load_checkpoint(filepath, pre_trained_model, training=False):
	'''
	Function: Load model from .pth file for training or predicting.

	Inputs:
	filepath - location of .pth file
	pre_trained_model - model used for image classification
	training - boolean to check whether optimizer needs to be loaded
	for further training

	Return: Classifer class with model loaded
	'''

	checkpoint = torch.load(filepath)

	classifier = Classifier(model=pre_trained_model,
		num_classes=checkpoint['num_classes'],
		class_mapping=checkpoint['mapping'],
		hidden_units=checkpoint['hidden_units'],
		dropout_p=checkpoint['dropout_p'],
		learnrate=checkpoint['learnrate'])

	classifier.model.load_state_dict(checkpoint['model_state_dict'])
	
	#If resuming training, load optimizer for backpropogation
	if training:
		classifier.optimizer.load_state_dict(checkpoint['optimizer'])

	return classifier

def process_image(image):
    ''' 
    Function: Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array as a PyTorch Tensor. Taken from Udacity Image Classifier
        project notebook.
	
	Inputs:
	image - PIL image
	
    Return: Image Tensor in PyTorch
    '''
    size = 256,256
    final_size = 224

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    with Image.open(image) as im:
        im.thumbnail(size)
        
        width,height = im.size

        #Center Crop Dimensions
        left = (width - final_size) // 2
        upper = (height - final_size) // 2
        right = (width + final_size) // 2
        bottom = (width + final_size) // 2

        im = im.crop((left, upper, right, bottom))
        
        #Make sure it scales to 224 x 224 after center crop
        im = im.resize((224,224))

        #Convert to Numpy array for color channel and normalization
        np_image = np.array(im)

        #Convert color channels encoded 0-255 to floats 0-1
        float_encoded = np.round(np.divide(np_image, 255), decimals=3)

        #Normalize to ImageNet Values
        float_encoded = (float_encoded - norm_mean) / norm_std

        #Re-order dimensions so the color channels are the first dimension
        final_array = np.transpose(float_encoded, (2,0,1))

        #Convert it to a PyTorch Tensor
        final_array = final_array.astype(np.float32)
        final_array = torch.from_numpy(final_array)

    return final_array

def imshow(image, ax=None, title=None):
    """
	Function: Show image from image tensor from PyTorch

	Inputs:
	image - PyTorch tensor of the image
	ax - Axes object if loading into a created plot figure
	title - Add title to the image produced using matplotlib.pyplot

	Return: Axes object containing the image plotted using matplotlib.pyplot
    """

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.axis('off')

    #plt.show()
    
    return ax

def predict(image_path, model, cat_to_name, device, top_k=5):
    ''' 
    Function: "Predict the class (or classes) of an image using a trained deep learning model."
    (Taken from the Udacity AI Programming Course Image Classifier Project Notebook)

    Inputs:
    image_path - image location
    model - Classifier model used
    cat_to_name - dict that contains class labels
    topk - how many classes probabilties to display for the prediction

    Return: numpy array of probabilities for k classes, numpy array of k class labels predicted
    '''

    image = process_image(image_path)
    image = image.to(device).unsqueeze(0)   #Processing only one image, add a batch size dimension
    model.to(device)
    
    model.eval()

    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_p, top_k_classes = ps.topk(top_k, dim=1)

    #Invert dictionary to map to indices to class labels
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}

    #Move top_p and top_k to cpu for numpy calculations
    cpu_top_k = top_k_classes.cpu().detach().numpy()
    cpu_top_p = top_p.cpu().detach().numpy()

    #Output class names predicted
    cat_names = [cat_to_name[idx_to_class[val]] for val in cpu_top_k[0]]

    print(f'Flower Name: {cat_names[0]}')

    return cpu_top_p[0], cat_names

def plot_p_and_k(top_p, predicted_labels, ax=None):
    ''' 
    Function: Plot top p probabilities and k class lables using matplotlib.pyplot to check image prediction.

    Inputs:
    top_p - numpy array containing the probabilities for k class labels
    predicted_labels - numpy array containing k class labels

    Return: Axes object plotting the probability graph for the prediction made
    '''
    
    if ax is None:
        fig, ax = plt.subplots()

    #fig, ax = plt.subplots()

    #cpu_top_p = top_p.cpu().detach().numpy()

    #Probability Graph
    ax.barh(np.arange(len(predicted_labels)), top_p, align='center')
    ax.set_yticks(np.arange(len(predicted_labels)), labels=predicted_labels)
    ax.invert_yaxis()
    ax.set_ylabel('Flower Category')
    ax.set_title("Probability Graph")

    #plt.show()

    return ax

class Classifier(nn.Module):
    '''
    Create a NN Image Classifier using a pre-trained model and class labels. Customize the number of hidden
    units and dropout of the feed-forward network. Note: Feed-forward architecture must match model trained,
    otherwise it will not be built correctly if loading a saved model.
    
    Inputs: 
    model - pre-trained model from torchvision.models
    classes - number of class labels for the data that is going to be trained on
    hidden_units - number of hidden units per layer in the feed-forward layer
    dropout_p - probability for the dropout layer
    class_mapping - index class labels for prediction
    '''
    def __init__(self, model, num_classes, class_mapping, hidden_units, dropout_p, learnrate):
        super().__init__()
    
        #Initialize variables
        self.num_classes = num_classes
        self.class_mapping = class_mapping
        self.hidden_units = hidden_units
        self.dropout_p = dropout_p
        self.learnrate = learnrate

        #Depending on architecture build off of pre-trained model
        if model == "resnet101":

            self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

            #Avoid backpropagation through pre-trained model
            for param in self.model.parameters():
                param.requires_grad = False

            #Feed-forward network for classifier -- consists of 2 linear layers, 1 RELU activation, dropout after RELU activation, and using LogSoftmax for class probabilities
            fc = nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(2048, self.hidden_units)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(p=self.dropout_p)),
                ('fc2', nn.Linear(self.hidden_units, self.num_classes)),
                ('output', nn.LogSoftmax(dim=1))
                ]))
            
            #Add feed-forward network to pre-trained model
            self.model.fc = fc

            #Add Loss function and Optimizer for Backpropogation
            self.criterion = nn.NLLLoss()
            self.optimizer = optim.AdamW(self.model.fc.parameters(), lr=self.learnrate)

        if model == 'vgg11':

            self.model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)

            #Avoid backpropagation through pre-trained model
            for param in self.model.parameters():
                param.requires_grad = False

            #Feed-forward network for classifier -- consists of 3 linear layers, 2 RELU activation, 2 dropouts after RELU activation, and using LogSoftmax for class probabilities
            fc = nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(25088, 4096)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(p=self.dropout_p)),
                ('fc2', nn.Linear(4096, self.hidden_units)),
                ('relu', nn.ReLU()),
                ('dropout2', nn.Dropout(p=self.dropout_p)),
                ('fc3', nn.Linear(self.hidden_units, self.num_classes)),
                ('output', nn.LogSoftmax(dim=1))
                ]))

            #Add feed-forward network to pre-trained model
            self.model.classifier = fc

            #Add Loss function and Optimizer for Backpropogation
            self.criterion = nn.NLLLoss()
            self.optimizer = optim.AdamW(self.model.classifier.parameters(), lr=self.learnrate)

        #Add indexed class labels
        self.model.class_to_idx = self.class_mapping
    
    def train(self, epochs, validation_interval, trainloader, valloader, device):
        '''
        Function: Take model from Classifier class and based on the training and validation datasets loaded, train the model
        based on the number of epochs the user inputted.
    
        Inputs:
        device - determine if GPU or CPU is used
        epochs - number of loops to train over
        validation_interval - at what interval is the validation loss calculated
        trainloader - training dataset DataLoader
        valloader - validation dataset DataLoader
    
        Return: None
        '''
        
        #Number of epochs and validation interval check
        epochs = epochs
        interval = validation_interval
        
        self.model.to(device)

        for e in range(epochs):
            training_loss = 0
        
            self.model.train()
    
            print(f'Epoch: {e+1}')
    
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
    
                logits = self.model.forward(images)
                loss = self.criterion(logits, labels)
                loss.backward()
    
                self.optimizer.step()
                self.optimizer.zero_grad()
    
                training_loss += loss.item()
    
            print(f'Training loss: {training_loss/len(trainloader):.3f}')
            
            if (e+1) % validation_interval == 0:
                test_loss = 0
                accuracy = 0
    
                self.model.eval()
    
                with torch.no_grad():
    
                    for images, labels in valloader:
                        images, labels = images.to(device), labels.to(device)
            
                        log_ps = self.model.forward(images)
                        loss = self.criterion(log_ps, labels)
                        test_loss += loss.item()
    
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f'Test Loss: {test_loss/len(valloader):.3f}',
        			f'Accuracy: {accuracy/len(valloader)*100:.3f}%')

    def test_model(self, acc_threshold, testloader, device):
        '''
        Function: Test Model for a acc_threshold accuracy pass using a test dataset separate
        from the training and validation datasets used for training.

        Inputs:
        acc_threshold - float denoting passing accuracy percentage in ff.ff% format.
        testloader - DataLoader with test dataset images

        Return: None
        '''

        accuracy = 0

        self.model.to(device)
        
        self.model.eval()
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                
                log_ps = self.model.forward(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            test_acc = accuracy/len(testloader)*100

            #Pass threshold set when method is called
            if test_acc >= acc_threshold:
                print(f'Accuracy: Pass with {test_acc:.2f}%')
            else:
                print(f'Accuracy: Fail with {test_acc:.2f}%')

    def save_model(self, filename, save_dir, classes, hidden_units, dropout_p, learnrate):
        '''
		Function: Save model to resume training at another timepoint as a filename.pth file in save_dir directory. 
		Ensure all information of the NN Architecture is captured.

		Inputs:
		save_dir: string of folder to save filename.pth file in
		classes - number of class labels
		hidden_units - number of hidden units for the linear layer
		dropout_p - probability of the dropout layer after RELU activation
		learnrate - learnrate used for the model

		Return: None
		'''

        #Ensure model includes dropout layer
        self.model.train()
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'mapping': self.model.class_to_idx,
            'hidden_units': self.hidden_units,
            'dropout_p': self.dropout_p,
            'learnrate': self.learnrate,
            'optimizer': self.optimizer.state_dict()}
        
        torch.save(checkpoint, save_dir + '/' + filename + '.pth')