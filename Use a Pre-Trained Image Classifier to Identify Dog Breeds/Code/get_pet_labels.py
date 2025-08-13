#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Arjun Sree Manoj
# DATE CREATED: 06/06/2025                  
# REVISED DATE: 06/08/2025
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    image_file_list = listdir(image_dir) # Get list of images from images folder
    results_dic = dict()                 # Initialize the dictionary

    for image_filename in image_file_list:
      filename = image_filename.lower().strip().split("_") # Re-format filename to lowercase, remove leading/trailing whitespace, split it by underscores
      pet_label = ""
      
      # Add only the dog name to the label
      for word in filename:
        if word.isalpha():
          pet_label += word + " "
      
      # Check if no duplicate image is added. If yes, show a warning.
      if image_filename not in results_dic.keys():
        results_dic[image_filename] = [pet_label.strip()]
      else:
        print(f"Image file {image_filename} already added.")
    
    # print(len(results_dic)) to check there were 40 entries in the dictionary.

    # Replace None with the results_dic dictionary that you created with this
    # function
    return results_dic
