# CS_T0828_HW1

Brief introduction:
	The task of this homework is car brand classification. I train a CNN by using transfer learning. And I also do some preprocessing/augmentation to improve the performance. The score finally I get in kaggle is 0.9296 .

Methodology:
	I choose the “ResNext101” as my model, and I didn’t fix any layer of the network. I just use the pre-trained model and retrain it. Compared with ResNet, they have same numbers of parameters,but the result is better. One 101-layer ResNext network has the same accuracy as a200-layer ResNet,but the former is only half the amount of calculation. 
	First,I resize the image to 256*256 as my iuput,and about preprocessing I used random horizontal flip with probability 0.5 and random rotation with 7angle degree,
and the epoch I run is 20 times.
	The input image normalized using mean = [0.485,0.456,0.406] and std = [0.229,0.224,0.225].The loss function is CrossEntropy and the optimizer is SGD.The hyperparameters is shown as following table.

Hyperparameters	Batch size	 Learning rate	  Momentum
Value	          10	         0.001	          0.9

Table 1.  The value of hyperparameters
