import NN_functions as nnf
import os

def main():
	'''
    Load data and create NN model for Image Classification. Then, train the model, test it for a 70% pass rate, and then save it as a .pth file
    to be used for predictions.
    
    Inputs: 
    Command-Line Arguments from nnf.get_input_train() method

    Output:
    Saves model as a .pth file.
    '''

	input_args = nnf.get_input_train()

	#Set to CPU or GPU depending on command-line arguments
	if input_args.gpu:
		device = nnf.device()
	else:
		device ="cpu"

	#Set Data Directory
	data_dir = input_args.data_dir

	#Initialize model architecture variables
	hidden_units = input_args.hidden_units
	learnrate = input_args.learning_rate
	epochs = input_args.epochs
	
	#Fixed Values for Model
	validation_interval = 10
	dropout_p = 0.2

	#Obtain datasets from data directory, load them into DataLoaders for NN training, and index class labels
	trainloader, valloader, testloader, class_mapping = nnf.load_data(data_dir)
	print("Loaded Training and Validation Datasets into DataLoaders for NN Training...")

	#Provide dictionary with class labels and number of class labels
	cat_to_name, num_of_classes = nnf.load_categories("cat_to_name.json")
	print("Dictionary with Class Labels Loaded...")

	
	#Build NN model with unique feed-forward layer
	classifier = nnf.Classifier(model=input_args.arch,
		num_classes=num_of_classes,
		class_mapping=class_mapping,
		hidden_units=hidden_units,
		dropout_p=dropout_p,
		learnrate=learnrate)

	print("NN Model Built...")

	print("Training Model...")
	#Train NN Model
	classifier.train(epochs, validation_interval, trainloader, valloader, device)
	print("Model Trained.")
	
	#Test Model
	print("Testing Model for a 70% pass rate...")
	accuracy_threshold = 70.00
	classifier.test_model(accuracy_threshold, testloader, device)

	
	#Save Model
	classifier.save_model('checkpoint', input_args.save_dir, num_of_classes, hidden_units, dropout_p, learnrate)
	print("Model Saved")

#Call to main to run the program
if __name__ == "__main__":
	main()