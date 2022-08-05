# CheXNet-with-localization


> **Note**
> The modify code that support input single DICOM and get multiple DICOM GSPS results from bounding box results
> Add features:
> - Input single DICOM and output multiple DICOM GSPS results
> - Extract inference into function that can be reusable (only support CPU mode)
> - Web API:
>   - GET: http://aimodel.example.com/ai_exec?filename=/home/user/chest.dcm
>   - POST: http://aimodel.example.com/ai_exec, body = form data => {file: file}, return zip file
>   - POST: http://aimodel.example.com/ai_exec_multipart, body = form data => {file: file}, return multipart/related

ADLxMLDS 2017 fall final

Team:XD

*黃晴 (R06922014), 王思傑 (R06922019), 曹爗文 (R06922022), 傅敏桓 (R06922030), 湯忠憲 (R06946003)*
## Weakly supervised localization :
In this task, we have to plot bounding boxes for each disease finding in a single chest X-ray without goundtruth (X, Y, width, height) in training set. The workflow is shown below:
### Workflow :
<img src="https://github.com/thtang/CheXNet-with-localization/blob/master/output/process_flow.png">
1) Predict findings
2) Use the classifier to plot heatmap (Grad-CAM)
3) Plot the bounding box base on Grad-CAM

### Package : 
- Python=3.6
The package changes that mention from [#8](https://github.com/thtang/CheXNet-with-localization/issues/8)
- ~~`Pytorch==0.2.0`~~ -> `pytorch==1.0.1`
```sh
# please using the anaconda
conda install pytorch==1.0.1 -c pytorch
```
- ~~`torchvision==0.2.0`~~ -> `torchvision==0.2.2`
- `matplotlib` 
- `scikit-image==0.13.1` 
- `opencv_python==3.4.0.12`
- `numpy==1.13.3` 
- `matplotlib==2.1.1` 
- `scipy==1.0.0` 
- `sklearn==0.19.1` 


### Environment:
* OS: Linux
* Python 3.5
* GPU: 1080 Ti
* CPU: Xeon(R) E5-2667 v4
* RAM: 500 GB
### Experiments process:
1) preprocessing:
```
python3 preprocessing.py [path of images folder] [path to data_entry] [path to bbox_list_path] [path to train_txt] [path to valid_txt] [path of preprocessed output (folder)]
```

2) training:
```
python3 train.py [path of preprocessed output (folder)]
```

3) local testing:
```
python3 denseNet_localization.py [path to test.txt] [path of images folder]
```
4) Output txt format: <br>
After running denseNet_localization.py, you would get a txt file. The format is shown below:<br>
```
[image_path] [number_of_detection]
[disease] [x] [y] [width] [height]
[disease] [x] [y] [width] [height]
...
[image_path] [number_of_detection]
[disease] [x] [y] [width] [height]
[disease] [x] [y] [width] [height]
...
```
------
For DeepQ platform testing:

upload **deepQ_25.zip** to the platform. Then use following command:
```
python3 inference.py
```
------

5) For **visualization**, please refers to [issue](https://github.com/thtang/CheXNet-with-localization/issues/9). Credit to [Sadam1195](https://github.com/Sadam1195).

### Get bounding box from trained model
From: [#7](https://github.com/thtang/CheXNet-with-localization/issues/7)
- Using the `denseNet_localization.py` in root path
```sh
python3 denseNet_localization.py [path to test.txt] [path of images folder]
```
- Where `test.txt` structure below; list the image filename that you want to get bounding box
```txt
xxx.jpg
uuu.jpg
yyy.jpg
```
- Where `path of images folder` is the image files' folder, e.g. /mnt/d/images
- The code will process the every line in test.txt and combine the images folder like `/mnt/d/images/xxx.jpg`
- Result's structure will be `{pathologies} x1 y1 x2 y2`
### Note :
In our .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
### Model :
<img src="https://github.com/thtang/CheXNet-with-localization/blob/master/output/multi_label_denseNet.png">
* Image is modified from Ref [2].

### Result :
*Prediction*<br>
<img src="https://github.com/thtang/CheXNet-with-localization/blob/master/output/prediction.png" width="320"><br>
*Heatmap per disease*
![Alt Text](https://github.com/thtang/CheXNet-with-localization/blob/master/output/heatmap_per_class.jpg)
Visualization of some heat maps with its ground-truth label (red) and its prediction
(blue) selected from each disease class. (From top-left to bottom: Atelectasis, Cardiomegaly,
Effusion, Infiltration, Mass, Nodule, Pneumonia and Pneumothorax)

*Bounding Box per patient*
![Alt Text](https://github.com/thtang/CheXNet-with-localization/blob/master/output/bb_select.JPG)
Visualization of some images with its ground-truth label (red) and its prediction
(blue) selected from each disease class.

**Refers to the [report](https://github.com/thtang/CheXNet-with-localization/blob/master/report.pdf) for more experiment results.**
## Reference:
1. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases* [[Arxiv]](https://arxiv.org/pdf/1705.02315.pdf)
2. *LEARNING TO DIAGNOSE FROM SCRATCH BY EXPLOITING DEPENDENCIES AMONG LABELS* [[Arxiv]](https://arxiv.org/pdf/1710.10501.pdf)
3. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning* [[Arxiv]](https://arxiv.org/pdf/1711.05225.pdf)
4. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* [[Arxiv]](https://arxiv.org/pdf/1610.02391.pdf)

## Contact:
Feel free to contact me (thtang@nlg.csie.ntu.edu.tw) if you have any problem.
