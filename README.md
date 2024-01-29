# Wind-Power-Generation-Forecasting
Wind power generation forecasting of a wind turbine using MLP(Multi Layer Perceptron)

## Dataset
In this project, we plan to utilize the real-time SCADA (Supervisory Control and Data
Acquisition) data from Kaggle [1]. The SCADA system serves as a vital link, connecting
turbines, substations, and meteorological stations to a central hub. This comprehensive
dataset, spanning from January 1, 2019, to December 14, 2021, is pivotal for our analysis.
Each entry corresponds to a 10-minute interval. The "Power Table" records power generation
measurements (Power(kW)) from a wind turbine, while the "Features Table" logs 77
attributes, including gearbox temperature, tower acceleration, torque, power factor, and tower
base temperature [1]. This rich set of attributes provides a multifaceted view of the turbine's
operation. Overall, the dataset comprises 136,730 data points [1].


## Preprocessing, Feature Engineering & Dataset Split

  In this project phase, we manipulate and enhance the dataset to
improve the predictive power of machine learning models. The process begins by merging two
datasets, 'features' and 'power,' based on timestamps, ensuring each feature corresponds to the
correct power output reading. To normalize the data and bring different scales to a comparable
range, two scaling techniques are employed: StandardScaler for standardizing the data
(transforming to have a mean of zero and a standard deviation of one) and MinMaxScaler for
scaling all features to a range between -1 and 1. This normalization is crucial for models
sensitive to the input data's scale. This step was carried out to improve our machine learning
algorithms' efficiency, stability, and performance, ensuring the training of more robust and
accurate models.

  Temporal relationships within the data are captured by adding sinusoidal and cosinusoidal
transformations of time-related attributes, like hours, months, and days. These transformations
are key in time series analysis, helping models understand and leverage cyclical patterns inherent
in time-based data. The core of feature engineering lies in creating lagged features and rolling
statistical features. Lagged components are generated to include past information, essential in
time series forecasting, where past values are strong predictors of future values. The dataset is
augmented with rolling mean and standard deviation calculations over a window of 6 time
periods for these lagged features. These rolling statistics provide insights into recent trends and
volatility in the data, which are often essential indicators in time series forecasting. Finally, the
dataset is tidied up by removing unnecessary columns and ensuring that the target variable,
'Power(kW),' is appropriately positioned. This results in a dataset rich with original and
engineered features tailored to enhance the project's subsequent predictive modeling
performance. Then, we again checked the absolute correlation of the new feature set:

![image](https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting/assets/99555923/03366b0f-b58e-4741-ab88-252375568672)

Figure 1: Sorted Absolute Correlation Between the Target(“Power(kW”)) and New Features

  After that, in the feature engineering step, we created lag features of the time series, lags of
rolling mean, and std to exploit the time series properties of the data. We then split our data of
136,730 samples into three splits: a training split of 109,375 data points, which is 80% of the
dataset, a validation split of 13,672 data points, and a test split of 13,672 data points, which are
both 10% of the dataset. We made sure to use these exact splits in the training, validating, and
testing of each of our machine-learning models to ensure an accurate evaluation of our different
models.

## MLP

  Multi-layer perceptron, in other words, a feed-forward neural network, is a neural network
consisting of the input layer, output layer, and hidden layer. There can be one or more hidden
layers in an MLP. The input layer consists of the input values we give to our model.

![image](https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting/assets/99555923/bbe75794-bfc3-43a3-92ec-582949190155)

Figure 2: A Basic Feed-Forward Neural Network. [2]

  Obtaining an output from the inputs in a neural network is called forward propagation. In
forward propagation, the linear combination of the weights of that neuron and the values of the
previous neurons is taken. After taking the linear combination, the result is passed through a non-
linear activation function, and the result of the activation function is the value of the neuron that
is passed to the next layer. We continue doing this until we obtain the output from the output layer.

![image](https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting/assets/99555923/b48bfa99-15b1-424f-8c0a-dd805126d260)

Figure 3: Architecture of a Simple MLP. [3]

  Improving our model based on the predictions a model made using forward propagating
the inputs and the actual result from a dataset is called training the model. The improvement in the
model is done by adjusting the weights in the neural network to get the predictions as close to the
actual values. We use backpropagation to adjust the weights in our neural network.
Backpropagation is a method that uses gradients to adjust weights.
  The gradient of a function at point P gives us the direction of increase in the value of the
function and the magnitude of the change at point P in the multi-dimensional space. We use the
gradient concept to reduce the errors we made in our predictions. We use the mean square error
(MSE) as our loss function that defines how close our predictions are to the actual values:

![image](https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting/assets/99555923/e5e65f1f-9e3b-4252-bda5-158ec91c8a50)

  Taking the gradient of the loss function will give us the direction and magnitude of the
increase of our error in the multi-dimensional weight space. Adding the negative gradient to our
weights will change the weights in the direction that reduces the error and change each weight
proportionally to the decrease they cause in the error. We repeat this backpropagation method
multiple times to obtain weights that give us a small error and thus make predictions close to the
actual values.
  We calculate the error gradient with respect to a weight using the chain rule. The use of the
chain rule in the primary network given in Figure 1 with the error function over 1 example (n=1)
is given below:

![image](https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting/assets/99555923/aa17bdbe-56d1-403b-aa0f-3e8792de3d5f)


Where f(z) is the activation function, and z is the linear combination of the weights and the inputs. 

![image](https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting/assets/99555923/077b9148-1fdc-45eb-ab58-8d6fa6fed081)


  Likewise, we use the chain rule to calculate the error gradient with respect to each weight
in the network, and we decrease the gradient from the weights using a constant learning rate.

![image](https://github.com/EnesAgirman/Wind-Power-Generation-Forecasting/assets/99555923/804766c3-bd5f-44ee-9067-f493109394e3)

  In our model, we use mini-batches to train our data, taking a batch of training data and
going through the batch to update the parameters after going through the batch. The advantage of
this method is that it performs more efficiently in a computer using parallel computing when
computing the gradients.

  We train our model with our training dataset that contains multiple input and output pairs.
We use forward propagation to find a prediction and use backpropagation to improve our model
on the training dataset. Going through the entire training dataset once is called one epoch. After
each epoch, the model’s accuracy is measured on a validation dataset separate from the training
dataset. We train our model for multiple epochs to improve it. After we train our model, we use a
test dataset separate from the train and validation datasets to test our model.


## References
[1] “Wind Turbine Power (kW) Generation Data,” Kaggle, 2019. [Online]. Available:
https://www.kaggle.com/datasets/psycon/wind-turbine-energy-kw-generation-data

[2] Kein, R. (2019) How to train a multilayer perceptron neural network, All About Circuits.
Available at: https://www.allaboutcircuits.com/technical-articles/how-to-train-a-multilayerperceptron-neural-network/ (Accessed: 25 December 2023). 

[3] NEURAL NETWORK (no date) Cosmos. Available at:
https://www.cosmos.esa.int/web/machine-learning-group/neural-network-introduction
(Accessed: 25 December 2023). 


