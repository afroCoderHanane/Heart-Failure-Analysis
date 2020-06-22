# Heart-Failure-Analysis

*refer to post on medium at : https://medium.com/@gbadhanane/pytorch-linear-regression-model-to-predict-life-tendency-of-people-with-heart-failure-clinical-e79274ede624

## 1-Abstract
According to the Center for Disease Control and Prevention(CDC), cardiovascular diseases kill approximately 647000 persons in the United States every year or one person every 37 seconds. These diseases mainly exhibit as myocardial infarctions and heart failures. Heart failure occurs when the heart cannot pump blood to meet the need of the body. Using a dataset of 299 people with heart failure, the goal is to predict, using machine learning, the chances of survival of an individual. I will be using PyTorch to build and train a model which task will be to predict if a subject will more likely die when he has a heart failure using the data in the column ejection fraction and serum creatinine.
Prior research has shown that the ejection fraction and serum creatinine provide more accurate data than if all the information that is in the different columns is used.[See table below]
Article about the efficiency if using ejection fraction and serum creatinine:
https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5
## 2- Introduction to Pytorch
Before moving to the analysis of the dataset of heart failure clinical records, it is important to know what is PyTorch. Pytorch is a machine learning framework based on torch library used for purposes such as Natural Language Processing, Computer Vision among others. Pytorch uses tensors and tensors can be numbers, vectors, matrix, or 3-D array. Tensor requires the data to be in regular shape meaning the number of elements in a 3-D array needs to be congruent.
Example of PyTorch tensor
Number: t1 = torch.tensor(1.0)
Vector: t2 = torch.tensor([1.,2,3,4])
## 3-Heart failure clinical record

heart_failure_clinical_records_dataset.csv
Plot



These graphs show that when the serum creatinine(<4) and the ejection fraction are lower (<60) the probability that the individual dies around 52%. 48% of the time he will survive.
## 4- Analysis
To successfully analyze the data, we need to prepare the data as the information that is brought into the program is stored as a CSV file. It needs to first be converted to a NumPy array and then to a torch tensor. Numpy arrays are different than torch tensor. Therefore, since I am using Pytorch converting a CSV data to a NumPy array will be done using to_numpy() function following the code below:

This process will transform the information read from the heart_failure_clinical_records_dataset.csv to NumPy arrays split into two which are the targets_array and the input arrays. Then the NumPy arrays need to be converted to tensors using from_numpy()

The tensors in their turn need to be split into two: the training set and the validation set of data. The ideology behind this split is to have two datasets that can be used to verify the accuracy or prove the efficiency of my model that will be created.

After the data is prepared for analysis, I build my model using the following pattern
Train the model: this model contains 3* nn.Linear classes that apply a linear transformation to the data. Thee is no better way to think about it that a function of weighted sum +bias. This gives a particular shape to the model because at the first application of the nn.Linear(),themodel is taken from the input size (12 )to 6 and applies the relu() function which particularity is to modify our model so that it is not a simple linear regression. The relu() function also get rid of the negative predict tendency of our model. Then, at the training, some predictions are generated and the loss is calculated and returned.
Validate the model using the validation: similarly to the last step of the training, the validation generates some predictions, applies a loss function, and returns the loss value.
Test the model: the testing is done at the end using batch value to generate predictions.

This process will generate a pretty much high-value loss after the model is initiated using model = LifeModel(). In this specific case the evaluate function returns a val_loss of : {‘val_loss’: 832.421875}.
So, my aim is to fit the program so that we have a small value loss as it will increase the accuracy of the model. The fit function uses a Stochastic optimization called Adam as an optimizer which is an algorithm that can be used in replacement of the Stochastic Gradient Descent.

After fitting the model several times using a high learning rate, at the first run the val_loss improve 80 times going from 800 to 10. After 2 runs of the fit functions using different learning rates, the value loss became 0.0952, which is below 0.1.
## Training
   refer to post on medium: https://medium.com/@gbadhanane/pytorch-linear-regression-model-to-predict-life-tendency-of-people-with-heart-failure-clinical-e79274ede624 

# Work Cited Pages
Pytorch series courses on youtube: https://www.youtube.com/watch?v=vo_fUOk-IKk&t=1s
zerotogans.com
PyTorch documentation - PyTorch master documentation
Learn about PyTorch's features and capabilities
pytorch.org
Dataset: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
