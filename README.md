# BinaryClassifier
This cat classifier takes an image as an input and then based on regression techniques , it predicts whether the image contains a cat or not with 70% accuracy and the tools used will be Jupyter Notebook and the code is written in python.

The main steps for building the classifier are: 
Define the model structure (such as number of input features) 
Initialize the model’s parameters 
Loop:

- Calculate current loss (forward propagation)
- Calculate current gradient (backward propagation)
- Update parameters (gradient descent)


The model comprises of :

- A training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- A test set of m_test images labeled as cat or non-cat
- Each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px). num_px = 64, m_train = 209 (Number of training examples) and m_test = 50 (Number of test examples).
