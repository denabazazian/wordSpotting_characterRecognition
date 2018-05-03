# wordSpotting_characterRecognition
A Word Spotting Method in Scene Images based on Character Recognition

## Code
The code is containing a CAFFE model for training an FCN to recognize characters in the scene images and an application of it for word-spotting. <br/>
### CAFFE
An FCN training for character detection (38 classes- 26 english letters incase sensetive, 10 numbers and signs) <br/>
The set up for training the net based on the CAFFE framework is at ```caffe/net_train/synth-fcn8s-atonce/```, and ```solv.py``` is for training. <br/>
### Model 
The CAFFE trained model of character recognition is at ```caffe/snapshot/```. <br/>
### Testing
In order to get the heatmaps, there are three python files at ```caffe/get_heatmap/```. The ```getHeatmaps.py``` is for visualizing the heatmaps of a list of images. ```getHeatmapsSINGLE.py``` is for visualizing the heatmap of a single image. ```saveHM.py``` is for saving the heatmaps of the images as png files. <br/>
#### Word Spotting
To try wordspotting based on these character recognition, there are some quantitative and qualitative files at ```caffe/word_spotting/```. <br/>

## Citation
Please cite this work in your publications if it helps your research: <br />

@InProceedings{Bazazian18, <br />
  author = {D.~Bazazian and D.~Karatzas and A.~Bagdanov}, <br />
  title = {Word Spotting in Scene Images based on Character Recognition }, <br />
  booktitle = {Proceeding of Computer Vision and Pattern Recognition Workshops}, <br />
  publisher = {IEEE}, <br />
  pages = {1-3}, <br />
  year = {2018} <br />
}

