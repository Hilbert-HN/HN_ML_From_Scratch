# HN_ML_From_Scratch
This is a repository to record the ML projects and useful information in my self-learning path on Machine Learning

**Also refer to my other AI Projects:** \
HN_Reinforcement_Learning_Projects: https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects

# TensorFlow Exercises
| TensorFlow Exercise | Description | Image | model.compile() | Last Activation Layer|
| ------------------- | ----------- | ----- | --------------- | -------------------- |
| 01-MNIST | Multiclass Classification for images of hand written digits  | ![image](https://user-images.githubusercontent.com/40123599/170816078-14dfc2e2-9f5d-455c-a310-0ba33d47b9dd.png) | optimizer='adam' or 'rmsprop', <br />loss='sparse_categorical_crossentropy',<br />metrics=['accuracy'] | softmax | 
| 02-Fashsion MNIST | Multiclass Classification for images of clothing |![image](https://user-images.githubusercontent.com/40123599/170819065-2cbcef21-973a-43dc-93cc-d7f04d4f0426.png)|optimizer='adam',<br />loss='sparse_categorical_crossentropy',<br />metrics=['accuracy'] | softmax |
| 03-IMDB Sentiment Analysis| Binary Classification for text of Movie Review |![image](https://user-images.githubusercontent.com/40123599/172426399-7f776100-8b02-49fc-aed0-9c3cd3039d96.png)| optimizer='adam',<br />loss=losses.BinaryCrossentropy(from_logits=True),<br />metrics=tf.metrics.BinaryAccuracy(threshold=0.0) | sigmoid |


# Typical Machine Learning Workflow
### Step 0 - Import Dependencies
<details>
  <summary>Common Dependencies</summary>
  
  **Tensorflow**
  <pre>
  import tensorflow as tf
  print(tf.__version__)
  </pre>
  
  **Keras**
  <pre>
  from tensorflow import keras
  from tensorflow.keras import layers
  
  #Optional
  from tensorflow.keras import losses
  </pre>
  
  **Numpy**
  <pre>
  import numpy as np
  </pre>
  
  **Matplotlib**
  <pre>
  import numpy as np
  from matplotlib import pyplot as plt
  
  #same as
  #import matplotlib.pyplot as plt
  </pre>
  
  **Handing Directory**
  <pre>
  import os
  import shutil
  
  #Refer to 03_IMDB_Sentiment_Analysis.ipynb
  </pre>
 
</details>

<details>
  <summary>Specific Dependencies</summary>
  
  **Handling pattern and text**
  <pre>
  import re
  # https://docs.python.org/3/library/re.html
  import string
  # https://docs.python.org/3/library/string.html
  
  #Refer to 03_IMDB_Sentiment_Analysis.ipynb
  </pre>
  
</details>

### Step 1 - Load Dataset
### Step 2 - Preprocessing the dataset
### Step 3 - Build the machine learning model
<details>
  <summary>The basic building block of a neural network is the layer. Layers extract representations from the data fed into them.</summary>
  
  **Example**
  <pre>
  model = keras.Sequential([
                            layers.Flatten(input_shape = (28,28)),
                            layers.Dense (128, activation = 'relu'),
                            layers.Dense(10)
  ])
  </pre>
</details>

### Step 4 - Compile Model with optimizer, loss, metrics
<details>
  <summary>Before start training a model, we need to pick an optmizer, a loss, and some metrics</summary>
  
  **Example**
  <pre>
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  </pre>
  
  
  
  <details>
    <summary>Optimizer</summary>
    This is how the model is updated based on the data it sees and its loss function.
  </details>
  
  <details>
    <summary>Loss</summary>
    This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
  </details>
  
  <details>
    <summary>Metrics</summary>
    Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
  </details>

</details>

### Step 5 - Train the model
### Step 6 - Evaluate the model
### Step 7 - Prediction


# Optmization vs Generalization


# Recommended Reference
**Book** \
[Deep Learning with Python, 2nd Edition (Manning Publications) - By François Chollet](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff) \
Github: https://github.com/fchollet/deep-learning-with-python-notebooks
