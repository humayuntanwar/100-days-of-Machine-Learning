# Machine-Learning
A Repository for Machine Learning Algorithms for easy Understanding
# A 100 day Machine Learing code pledge

### Day 1
  started from learing regression algorithm where we used the wiki/googl data of stocks to display data.
  ##### Libraries used: pandas ,quandl
  
### Day 2
  on day two we used the same data with labels and features and did some calculations on the data we were getting.
  ##### Libraries used:+ maths
  
### Day 3
  first time used linear Regression to test and train using linear_model from Linear Regression.
  ##### Libraries used : + numpy , sklearn
  
### Day 4
  firstly,we used regresion for forecasting and predicting the the future stock, also plotted the values
  ##### Libraries used : + datetime , matplotlib and style from mpl 
  
### Day 4 and 5 
  completed Siraj rivals linear Regression using Gradient Descent live stream work along to predict more practices mean better results
  
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/LinearRegressionUsingGradientDescent/gradient_descent_example.gif)
  
  
### Day 5
  learned pickling, in which we save our classifer as a pickle file and then loaded the pickle
  ##### Libraries used :  + pickle
  implemented best fit line slope eqaution y= mx+b , also made predict of y based on xs, plotted the points and line 
  #### Libraries used = numpy + matplotlib
  
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/best-fit-line-plot.png)
  
  


### Day 6
  Testing Assumptions
  
  ![ALt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/img/testingassumtion.png)
  
  
  studied R Squared
  ![Alt Text](https://i.stack.imgur.com/xb1VY.png)
  

http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit

https://www.coursera.org/lecture/linear-regression-model/r-squared-lMej8

https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/r-squared-or-coefficient-of-determination
  Implemented R Squared

### Day 7
   Started Classification w/ K Nearest Neighbors
   https://www.youtube.com/watch?v=44jq6ano5n0&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=13
   
   https://www.youtube.com/watch?v=44jq6ano5n0&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=13

   Using the Breast Cancer Dataset from Wisconsin-Madison
   https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

   Implemented K nearest Neighbours on Breast cancer data
  ##### Libraries used: numpy , neighbors from sklearn, pandas

  Studied Euclidean Distance
  https://www.youtube.com/watch?v=hl3bQySs8sM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=15
  https://www.youtube.com/watch?v=imD_XsEV-90
  ![ALt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/img/Euclidean_distance.png)

  Implemted Euclidean distance formula
  
  Implented K nearest neighbor algorithm from scratch using our own data, classfied one point
  images also added to see differences
  
  ![ALt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/img/knearestunclassified.png) 
  ![Alt text](https://github.com/humayuntanwar/Machine-Learning/blob/master/img/knearestclassified.png)
  
  ##### Libraries used: numpy, matplotlib, warnings, sqrt,collections 

### Day 8

  Testing our own k nearest neighbor algorithm on breast cancer dataset accuracy of 95%

  comapared our k nearnest with numpy k nearest
  our accuracy is 96.7% , numpy is 97.2% :(

### Day 9

  Studied the theory of the Support Vector Machine (SVM), which is a classification learning algorithm for machine learning. We also show how to apply the SVM using Scikit-Learn on some familiar data.

  https://www.youtube.com/watch?v=mA5nwGoRAOo&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=20

  https://www.youtube.com/watch?v=_PwhiWxHK8o

  Understanding Vectors
  https://www.youtube.com/watch?v=HHUqhVzctQE&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=21
  https://www.youtube.com/watch?v=fNk_zzaMoSs

  Studied Support Vector Assertion
  https://www.youtube.com/watch?v=VngCRWPrNNc&index=22&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

  support vectors fundamentals
  https://www.youtube.com/watch?v=ZDu3LKv9gOI&index=23&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
  
### Day 10
  completed the implementation of Support Vector Machine (boy its messy)

  tested the SVM
  
 ### Day 11
  Completed the support vector Implementation , tested the algorithm and did some prediction on the test data
  also did visualization
  before after here 
  ##### Libraries used :  + numpy
  
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/SvmInitialtest.png)
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/classificatinwithprediction.png)

  
 ### Day 12
  studied indepth about kernals
  studied in depth about clustering
  implemented clustering from sklearn
	
  #### Libraries used : sklearn

  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/cluster-1.png)

  

 ### Day 13
  
  today i practiced the handling of non numeric values in data using the titanic data set.
  the approach we used was identifying unique elements in colums and asigning them values

  #### Libraries used : sklearn +pandas + sklearn,cluster

  passed the processed titanic data thorugh the kMeans clustering to see how many people survived
  and how many didnt getting fairly good prediction of about 70%

 ### Day 14
  implmented the K means clsutering algorithm from the scratch learned alot of the basic functionality

  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/centroids.png)

  tested our own k means on titanic data
  compared results with sklearn kmeans
  ours is fasters, maybe due to haevy sklearn imports


### Day 15
  Studied Means Shift
  https://www.youtube.com/watch?v=3ERPpzrDkVg&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=39

  Implemented the meanshift from numpy on the titanic data set.
  and saw which group of people survived the most
  conclusion: the ones who paid he most for their ticket survived

### Day 16
  implemented Means Shift Algorithm from scratch to get the full understanding 
  "with radius as 4 " for our easy of us
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/means_shift_r=4.png)

  implemented Means shift algorithm with a dynamic radius,
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/MSdynamicradius.png)

  implemented it as data from make blob of sklearn
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/plots/mswithblobn50.png)


### DAY 17
  STARTED LEARNING NEURAL NETWORKS (cool stuff)
  https://www.youtube.com/watch?v=oYbVFhK_olY&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=43
  ![Alt Text](https://github.com/humayuntanwar/Machine-Learning/blob/master/img/neural.png)

  Installed and setup of TensorFlow
  RAN TENSORFLOWWWWW

### DAY 18
  Studied further about the implementation of neural networks and deep learning

  modelled a neural network using the tensorflow library on the mnist data set
  ##### Libraires Used: Tensor Flow

### DAY 19
  trained and tested the nueral network on the mnist data set
  
  ##### Libraires Used: Tensor Flow

### DAY 20
  Started working on a project where we will be creating a:
  Setimental Analyzer to analyze postive and negative sentiments in our data set
  Using a deep nueral Netwrok
  today Installed Nltk and created a data set and did the imports 

  ##### Libraires Used: Nltk, Numpy, random ,pickle, collections

### DAY 21
  implemnted further our sentimented analyzer
  developed the lexicon
  saved lexicon as a pickle
  downloaded nltk libs
  
  ##### Libraires Used: Nltk, Numpy, random ,pickle, collections
