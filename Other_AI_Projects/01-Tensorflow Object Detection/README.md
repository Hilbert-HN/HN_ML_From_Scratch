# Project 01 - Tensorflow Object Detection

[Youtube Tutorial] - Nicholas Renotte
https://www.youtube.com/watch?v=yqkISICHH-U&t=11444s

[Git Hub Tutorial]
https://github.com/nicknochnack/TFODCourse

[LabelImg]
https://github.com/tzutalin/labelImg

[Tensoflow 2_detection zoo]
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

## File Structure
![image](https://user-images.githubusercontent.com/40123599/166134808-13df672b-e37d-46a0-af72-0a19e7872445.png)

## 1 - Image Collection via Camaera 
**[Skip this if using image readily availabe]**

<b>a - Image Collection.ipynb</b> [Jupyter Notebook]  \
https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects/blob/master/03_Other_AI_Projects/01-Tensorflow%20Object%20Detection/1.%20Image%20Collection.ipynb
<br/><br/>

## 2 - Image Labling
<b>a - Image Labeling Install.ipynb</b> [Jupyter Notebook]  \
https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects/blob/master/03_Other_AI_Projects/01-Tensorflow%20Object%20Detection/1.%20Image%20Collection.ipynb

<b>b - Nagivate to the labelmg files and run labelImg.py in Command Prompt</b> 
<pre>
cd TFODCourse\Tensorflow\labelimg
python labelImg.py
</pre>

<b>c - Start Labeling</b>\
Hotkeys: w - Create a rect box | d -  Next image | a - Previous image
![image](https://user-images.githubusercontent.com/40123599/166134374-97852a27-350b-496c-8aa3-0bde0a624f32.png)

## 3 - Transfer Images to Train & Test Files
<b>a - Transfer Images to Train & Test Files.ipynb</b> [Jupyter Notebook] \
https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects/blob/master/03_Other_AI_Projects/01-Tensorflow%20Object%20Detection/3.%20Transfer%20Images%20to%20Train%20%26%20Test%20Files.ipyn

<b>b - The zipped files of train and test to be uploaded to Colab</b> \
![image](https://user-images.githubusercontent.com/40123599/166135029-6f1c387b-5743-419b-b08a-8322298dbab9.png)

## 4 - Training and eval in Colab
<b>a - Training and Detection.ipynb</b> [Colab] \
https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects/blob/master/03_Other_AI_Projects/01-Tensorflow%20Object%20Detection/4.%20Training%20and%20Detection.ipynb

![image](https://user-images.githubusercontent.com/40123599/166135223-2d3bb087-2f8f-42a5-93fd-89666917d4cc.png)

## 5 - View Training and Evlalation Graph in TensorBoard 
<b>a - Training and Detection.ipynb</b> [Colab- After Step 7] \
https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects/blob/master/03_Other_AI_Projects/01-Tensorflow%20Object%20Detection/4.%20Training%20and%20Detection.ipynb

<b>b- Navigate to train / eval folder in command prompt</b> \
![image](https://user-images.githubusercontent.com/40123599/166990609-e0b8b6d6-d4a6-4c50-80a1-58a7ff104fd1.png)

<b>c - Run below command</b>
<pre>
tensorboard --logdir=.
</pre>

<b>d - Copy below link in browser</b>
<pre>
http://localhost:6006/
</pre>

<b>[Tensorboard Example]</b>
![image](https://user-images.githubusercontent.com/40123599/166994295-00db471e-c10a-497d-b4ad-edee5fe2d4d1.png)

## 6 - Import trained model
<b>a - Training and Detection.ipynb</b> [Colab - Skip Step 3-7] \
https://github.com/Hilbert-HN/HN_Reinforcement_Learning_Projects/blob/master/03_Other_AI_Projects/01-Tensorflow%20Object%20Detection/4.%20Training%20and%20Detection.ipynb \

<b>b - Copy below pipeline.config + 3 files in checkpoint folder from local drive</b> \
![image](https://user-images.githubusercontent.com/40123599/166971205-3e9b05ce-472b-40d2-8092-c6e8d68a15f2.png) \

<b>c - Paste in Colab</b> \
![image](https://user-images.githubusercontent.com/40123599/166971532-a1d62790-a931-4023-ab52-204c24ffa722.png) \

