### My AI Repositoriesssss
> [HN_ML_From_Scratch](https://github.com/Hilbert-HN/HN_ML_From_Scratch)  
> [HN_Reinforcement_Learning_Projects](https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects)  
> [Playground_AI](https://github.com/Hilbert-HN/Playground_AI)

# HN_ML_From_Scratch
This is a repository to record the ML projects and useful information in my self-learning path on Machine Learning

# TensorFlow Exercises
TensorFlow Tutorial: https://www.tensorflow.org/tutorials
| TensorFlow Exercise | Description | Image | model.compile() | Last Activation Layer|
| ------------------- | ----------- | ----- | --------------- | -------------------- |
| [01-MNIST](01_TensorFlow_Exercises/01_MNIST.ipynb) | Multiclass Classification for images of hand written digits  | ![image](https://user-images.githubusercontent.com/40123599/170816078-14dfc2e2-9f5d-455c-a310-0ba33d47b9dd.png) | optimizer='adam' or 'rmsprop' <br />loss='sparse_categorical_crossentropy'<br />metrics=['accuracy'] | softmax | 
| [02-Fashsion MNIST](01_TensorFlow_Exercises/02_Fashsion_MNIST.ipynb) | Multiclass Classification for images of clothing |![image](https://user-images.githubusercontent.com/40123599/170819065-2cbcef21-973a-43dc-93cc-d7f04d4f0426.png)|optimizer='adam'<br />loss='sparse_categorical_crossentropy'<br />metrics=['accuracy'] | softmax |
| [03-IMDB Sentiment](01_TensorFlow_Exercises/03_IMDB_Sentiment.ipynb) <br /> [04-IMDB Sentiment(with_TF_Hub)](01_TensorFlow_Exercises/04_IMDB_Sentiment(with_TF_Hub).ipynb)| Binary Classification for text of Movie Review |![image](https://user-images.githubusercontent.com/40123599/172426399-7f776100-8b02-49fc-aed0-9c3cd3039d96.png)| optimizer='adam',<br />loss='Binary_Crossentropy'<br />metrics=['binary_crossentropy'] | sigmoid |


# Typical Machine Learning Workflow (Keep Updating)
### Step 0 - Import Dependencies
<details>
  <summary>Common Dependencies</summary>
  
  **Tensorflow**
  ```python
  import tensorflow as tf
  print("Version: ", tf.__version__)
  
  print("Eager mode: ", tf.executing_eagerly())
  print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
  ```
  **Tensforflow_hub**
  ```python
  import tensorflow_hub as hub
  print("Hub version: ", hub.__version__)
  ```
  
  **Keras**
  ```python
  from tensorflow import keras
  from tensorflow.keras import layers
  
  #Optional
  from tensorflow.keras import losses
  ```
  
  **Numpy**
  ```python
  import numpy as np
  ```
  
  **Matplotlib**
  ```python
  import numpy as np
  from matplotlib import pyplot as plt
  
  #same as
  #import matplotlib.pyplot as plt
  ```
  
  **Handing Directory**
  ```python
  import os
  import shutil
  
  #Refer to 03_IMDB_Sentiment_Analysis.ipynb
  ```
 
</details>

<details>
  <summary>Specific Dependencies</summary>
  
  **Handling pattern and text**
  ```python
  import re
  # https://docs.python.org/3/library/re.html
  import string
  # https://docs.python.org/3/library/string.html
  
  #Refer to 03_IMDB_Sentiment_Analysis.ipynb
  ```
  
</details>

### Step 1 - Load Dataset
### Step 2a - Explore the data
```python
train_examples_batch, train_lables_batch = next(iter(train_data.batch(10)))
train_examples_batch
```
### Step 2b - Preprocessing the dataset
### Step 3 - Build the machine learning model
The basic building block of a neural network is the layer. Layers extract representations from the data fed into them.

<details>
  <summary>Differnet Ways to Build the model</summary>
  
  **Sequential Model**  
  
  There are 2 ways of building Sequential model  
  ```python
  model = keras.Sequential([
                            layers.Dense(64, activation = 'relu'),
                            layers.Dense(10, actuvation = 'softmax')
  ])
  ```
  ```python
  model = keras.Sequential()
  model.add(layers.Dense(64, activation = 'relu'))
  model.add(layers.Dense(10, activation = 'softmax'))
  ```
</details>

<details>
  <summary>Visualize the Model</summary>
  
  **Summary**  
  ```python
  Model.summary()
  ```
  ![image](https://user-images.githubusercontent.com/40123599/173235079-47d79fc4-9d6d-4812-8426-e353461ca38c.png)

</details>
  


### Step 4 - Compile Model with optimizer, loss, metrics

Before start training a model, we need to pick an optmizer, a loss, and some metrics

**Example**
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

<details>
  <summary>Optimizer</summary>
  This is how the model is updated based on the data it sees and its loss function.    
</details>

<details>
  <summary>Loss</summary>
  This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.<br /> 
  Tensorflow.keras.loss Documenation - https://www.tensorflow.org/api_docs/python/tf/keras/losses  
  <br /><br />
  There are 2 common ways for calling the loss functions with model.compile ( ) API

  **Recommended Usage (set from_logits=True)**
  ```python
  # y_pred represents a logit, i.e, value in [-inf, inf]  
  # Last activation layer is not included  
  
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
  ```

  **Default Usage (set from_logits=False)**  
  ```python
  # y_pred represents a probability, i.e, value in [0, 1]  
  # Last activation layer is included  
  
  loss='binary_crossentropy'
  ```

</details>

<details>
  <summary>Metrics</summary>
  Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
</details>

</details>

### Step 5 - Train the model
### Step 6 - Evaluate the model & Visualize the loss and accuracy
### Step 7 - Add Last Activation Layer 
### Step 8 - Prediction

# Recommended Reference
**Book** \
[Deep Learning with Python, 2nd Edition (Manning Publications) - By François Chollet](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff) \
Github: https://github.com/fchollet/deep-learning-with-python-notebooks

**Dataset**  
[COCO Common Objects in Context](https://cocodataset.org/#home)  
COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features

**Blog**  
[Complete Guide to Data Augmentation for Computer Vision](https://towardsdatascience.com/complete-guide-to-data-augmentation-for-computer-vision-1abe4063ad07)
