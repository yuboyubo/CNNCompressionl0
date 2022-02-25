1. Getting started

Prerequisites

	Python 3.6+
	PyTorch 1.4

Installation

	Python 3.x is installed by default on either Linux or Mac. To install PyTorch on Mac or Linux, you can run
	
		pip3 install torch torchvision

	If you have any problems with PyTorch installation, please visit 

		https://pytorch.org/get-started/locally/

	If you are using Windows or you would not like to install any packages on your computer, we highly recommend Google's Colab which is really handy. Then you'll get rid of any trouble about installation. If so, you might need to comment some code w.r.t parsing arguments. It's not complicated at all. You'll see some instructions in there.

2. Data

MNIST dataset is publicly available. When you run our code, the dataset will be downloaded automatically.

3. Reproducibility

After you download our source code, you'll see a folder named 'reproducible', which includes the test code and the pruned models. If you want to double check if we have achieved the reported test accuracy with the reported number of nonzero parameters, you can run our test code(reproducible.py) via the following commands. 

	cd reproducible
	python3 reproducible.py

The test code is pretty straightforward. We first load data and then use some simple methods provided by PyTorch, like '.nonzeros()'. Specifically, it contains 4 functions and each of them has only a couple of lines. For more info, you can read reproducible.py.

4. How to run our regular code

We pre-trained LeNet-300-100 and LeNet-5 on MNIST achieving test accuracy of 98.24% and 99.12%, respectively. Also, we provide the pre-trained models to make sure that we have the same starting points. You will see the following two pre-trained models in the source code folder.

	LeNet300_pretrained_0.9824.pkl
	LeNet5_pretrained_0.9912.pkl
	
There is one important argument to specify for our iterative DNN_PALM_L0 implementation, i.e. the shrinkage coefficient(lambda). However, if you just want to test our code with the reported arguments in our paper, please run the following commands in your terminal.

To prune LeNet-300-100,

	python3 LeNet300_L0_mnist.py

To prune LeNet-5,

	python3 LeNet5_L0_mnist.py

If you would like to try a different lambda, you can run the following command. * denotes the source code file you want to run and --l0 is the lambda you want to try.

	python3 *.py --l0 0.01

For our iterative DNN_PALM_L0_Group_Lasso implementation, there are two key argumetns, i.e. lambda(shrinkage strength for group lasso) and eta(shrinkage strength for L0 norm). There are two identical typos in our Eq. (21) and Eq. (22). The subscript of the second term is supposed to be 2, while the subscript of the third term should be 0. Sorry about that.

To prune LeNet-300-100 with sparse group lasso algo,

	python3 LeNet300_SparseGL_mnist.py

To prune LeNet-5 with sparse group lasso algo,

	python3 LeNet5_SparseGL_mnist.py

If you would like to try a different lambda, you can run the following command. --l0 denotes shrinkage strength on L0 norm and --l21 is the shrinkage strength on group lasso.

	python3 *.py --l0 0.0003 --l21 0.0002

If you want to test the non-iterative counterparts of our algorithms, you only need to comment the last line of our source code files, i.e. the following line 

	model = retrain(model, train_set, val_loader, wd, dnnepochs, step, device)# retraining process. 

5. More details

You can adjust more arguments when running our code. For example, you can specify the number of epochs for pruning or for retraining,

	--pruningepochs 10 	# pruning for 10 epochs
	--dnnepochs 30 	# retraining for 30 epochs

Also, you can increase the shrinkage strength by a certain increment.

	--inc 0.001 --stride 30 # to increase the shrinkage strength for L0 norm and group lasso per 30 iterations

Each iteration consists of 10 epochs pruning and 30 epochs retraining. 

Note that the implementation for the proximal operator of sparse group lasso, i.e. Algo 4 in our paper, might be not very straightforward, because we use matrix operations instead of for loops to speed up our program. We provide two versions for this part.	The function GroupSparse_sol() runs fast since we use matrix manipulations, while GroupSparse_sol_forloop() is pretty straightforward since it closely follows the formulas in our paper. If you run them with the same input, you'll get the same output.

There are intensive comments in our source code, particularly in LeNet300_L0_mnist.py, so please feel free to read our source code.

We are thankful that you can read through this readme file. Thanks so much for your time to review our paper.

