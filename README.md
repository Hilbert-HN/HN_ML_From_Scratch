# HN_ML_From_Scratch

| Machine Learning Exercise | Description | Image | Hyperparameters |
| ------------------------- | ----------- | ----- | --------------- |
| 01-MNIST | Calssify images of hand written digits  | ![image](https://user-images.githubusercontent.com/40123599/170816078-14dfc2e2-9f5d-455c-a310-0ba33d47b9dd.png) | optimizer='adam'<br />loss='sparse_categorical_crossentropy',<br />metrics=['accuracy']|
| 02-Fashsion MNIST | Classify images of clothing |![image](https://user-images.githubusercontent.com/40123599/170819065-2cbcef21-973a-43dc-93cc-d7f04d4f0426.png)|optimizer='adam'<br />loss='sparse_categorical_crossentropy',<br />metrics=['accuracy']|

## Step 0 - Import Dependencies
<details>
  <summary>Common Dependencies</summary>
  <pre>
    import tensorflow as tf
    print(tf.__version__)
  </pre>
  <pre>
    from tensorflow import keras
    from tensorflow.keras import layers
  </pre>
</details>

## model.compile()
Before start training a model, we need to pick an optmizer, a loss, and some metrics
### Optimizer
### Loss
### Metrics
