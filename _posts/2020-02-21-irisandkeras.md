---
title: "Iris and Keras"
date: 2020-02-21
categories: keras neural-networks ml classification tutorials
comments: true
---


Our goal: to use classification methods to use properties of observed flower measurements from the `iris` dataset to predict the type of iris flower species the measurements refer to.

**multi-class classification**: A classification problem that involves more than two classes to be predicted. 

Let's start by importing the packages and functions we will need for this. We are using `Keras`, a deep learning library, as well as `pandas`- which helps with data manipulation, and `scikit-learn`, which will help us with evaluating our model as well as preparing our data for modeling.


```python
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
```

    Using TensorFlow backend.


Next, we will load our data, and inspect it.


```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=names)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal-length</th>
      <th>sepal-width</th>
      <th>petal-length</th>
      <th>petal-width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



Let's split our attributes into input and output variables, to map a function X to Y.


```python
data = df.values
X = data[:,0:4].astype(float)
Y = data[:,4]
```


```python
df['class'].unique()
```




    array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)



We see we have 3 different class values. But, in order to model appropriately, we need to reshape our output attribute to become a matrix with a boolean for each class value, and also tells us whether or not a given observation has that class value or not. 

This is called **one-hot encoding**, also known as creating **dummy variables**.
This is used for categorical variables, not numerical.


How do we do this? We do this by firstly, encoding the strings consistently to integers using the `LabelEncoder` in scikit-learn. Then, we convert the resultant vector of integers to a one-hot encoding format using the Keras function `to_categorical()`.


```python
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
```

Next, we will go ahead and define our neural network model.

The Keras library provides us a multitude of classes to use neural network models.

We will make use of a `KerasClassifier` class, used as an estimator in scikit-learn.

We will create a baseline neural network to classify the type of iris flower species we have in our model, by creating a simple fully connected network with one hidden layer that contains 8 neurons.

The hidden layer uses an activation function called `relu`. The output layer must create 3 output values, one for each class. Why is this? This is because we used one-hot encoding for our iris data. The output value with the largest value will be taken as the class that is predicted by the model. 

So, we will have 4 inputs, 8 hidden nodes, and 3 outputs.

We will use a `softmax` activation function in the output layer - this is done so that we have output values between the range of 0 and 1, and these will be used as predicted probabilities.

The network uses the efficient gradient descent optimization algorithm called Adam - and we will also be using a logarithmic loss function, defined as `categorical_crossentropy`. 

For the scope of this tutorial, we won't go into depth just yet about why these were chosen.


```python
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

Now, we can build our `KerasClassifier` so we can use it in scikit-learn. 

Secondly, important to note, we will pass arguments in the construction of this class that will be passed internally to the fit( ) function, to train our neural network.

We will pass the number of epochs as 200, and our batch size as 5 during training.

We are also turning debugging off during training, by setting `verbose` to 0.


```python
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
```

Now, we can proceed to evaluate the neural network model we built on our training data. 

Scikit-learn can help us with evaluation - and the gold standard for evaluating machine learning models is using a technique known as k-fold cross validation.

First, we will define our model evaluation procedure. We are setting the number of folds to be 10 (a good default to go with), and we are shuffling our data before partitioning it - to reduce bias.


```python
kfold = KFold(n_splits=10, shuffle=True)
```

We can now evaluate our estimator model onto our dataset, using a 10-fold cross validation procedure. 

Evaluating the model will take around 10 seconds, and what happens is we are returned with an object that describes the evaluation of the 10 different constructed models for each of the splits in the data we have made.


```python
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

    Baseline: 88.67% (25.31%)


As we can see, our results are shown as both the mean and standard deviation of the accuracy of the model.

Essentially here, we have done the following:

- load the data, and made it available to the Keras package
- prepared multi-class classification data for modeling, using one-hot encoding procedures 
- used a Keras neural network model with scikit-learn
- defined a neural network using Keras for multi-class classification
- evaluated the neural network using scikit-learn with the use of k-fold cross validation

Hopefully, you have learned something from this in terms of an introductory look into using Keras, as well as the usage of a  basic machine learning workflow. 
