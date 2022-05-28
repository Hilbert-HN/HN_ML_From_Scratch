# HN_ML_From_Scratch

| Machine Learning Exercise | Description | Image | Hyperparameters |
| ------------------------- | ----------- | ----- | --------------- |
| 01-MNIST | Calssify images of hand written digits  | ![image](https://user-images.githubusercontent.com/40123599/170816078-14dfc2e2-9f5d-455c-a310-0ba33d47b9dd.png) | optimizer='adam'<br />loss='sparse_categorical_crossentropy',<br />metrics=['accuracy']|
| 02-Fashsion MNIST | Classify images of clothing |![image](https://user-images.githubusercontent.com/40123599/170819065-2cbcef21-973a-43dc-93cc-d7f04d4f0426.png)|optimizer='adam'<br />loss='sparse_categorical_crossentropy',<br />metrics=['accuracy']|

# Typical Machine Learning Workflow
## Step 0 - Import Dependencies
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
  </pre>
  
  **Numpy & Matplotlib**
  <pre>
  import numpy as np
  import matplotlib.pyplot as plt
  </pre>
  
</details>

## Step 1 - Load Dataset
## Step 2 - Preprocessing the dataset
## Step 3 - Build the machine learning model
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

## Step 4 - Train the model
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
  </details>
  
  <details>
    <summary>Loss</summary>
  </details>
  
  <details>
    <summary>Metrics</summary>
  </details>

</details>

## Step 5-Train the model
## Step 6-Evaluate the model
## Step 7-Prediction

