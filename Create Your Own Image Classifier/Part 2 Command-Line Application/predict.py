import NN_functions as nnf
import os
import matplotlib.pyplot as plt

def main():
	'''
	Load a trained model and predict classification of images.

	Inputs: 
	Command-Line Arguments from nnf.

	Output:
	Image plot and Probability Graph for Top k classes selected
	'''
	input_args = nnf.get_input_predict()

		#Set to CPU or GPU depending on command-line arguments
	if input_args.gpu:
		device = nnf.device()
	else:
		device ="cpu"

	#Provide dictionary with class labels and number of class labels
	cat_to_name, num_of_classes = nnf.load_categories(input_args.category_names)
	print("Dictionary with Class Labels Loaded...")

	#Test an Image
	test_image = input_args.path_to_image

	print("Loading Trained Model...")
	#Load Trained Model
	classifier = nnf.load_checkpoint('checkpoint.pth', "resnet101")
	print("Model Loaded...")

	print("Predicting Image Classifications...")
	#Find probabilities for top k (5) predicted class labels
	top_p, cat_names = nnf.predict(test_image, classifier.model, cat_to_name, device, top_k=input_args.top_k)

	#Create subplots
	fig, (ax1, ax2) = plt.subplots(2,1)
	fig.canvas.manager.set_window_title('Flower Image and Name Predication Probability Graph')
	

	#Plot flower image and probability graph
	image = nnf.process_image(test_image)
	ax1 = nnf.imshow(image, ax1)
	ax2 = nnf.plot_p_and_k(top_p, cat_names, ax2)

	#Show subplots
	plt.tight_layout()
	plt.show()

#Call to main to run the program
if __name__ == "__main__":
	main()