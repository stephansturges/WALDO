# WALDO V1.0
\
Whereabouts Ascertainment for Low-lying Detectable Objects (get it?) 


## What is this?
\
WALDO is a trained detection (bounding-box) deep neural network to enable overhead detection of land-based objects!
These are the current detection classes:

1 --> car \
2 --> van \
3 --> truck \
4 --> building \
5 --> human \
6 --> gastank \
7 --> digger \
8 --> container \
9 --> bus \
10 --> pylon \
11 --> boat \
12 --> bike \

This AI system is primarily designed to be used for ground-risk mitigation for large flying objects traveling over populated areas, but it can also be useful for all sorts of other things like "search and rescue"-type operations, disaster relief etc... it's up to you!

If you need ground-risk segmentation instead of object detection make sure to check out OpenLander here: https://github.com/stephansturges/OpenLander/

## How well does it work? 

Check out the video below for a high-level idea of the performance of the default 960px full-size model. \

[![Video of WALDO doing detection](https://img.youtube.com/vi/7AlyRft_GXw/0.jpg)](https://www.youtube.com/watch?v=7AlyRft_GXw)


Multiple other versions of the model are coming very soon, including (of course!) versions that are optimized to be embedded for real-time _ground-risk mitigation_. 

Some detection metrics from training: 
![confusion_matrix](https://user-images.githubusercontent.com/20320678/233322563-4770423b-6d97-4221-ae2e-c8c63961e6e3.png)


As you can see classes that are close like "car" and "van" suffer from some confusion. Same goes for things where you often require more context to understand the object like "truck" / "container" / "gastank"... and some classes require more data / just more training to get better. Feel free to donate with the Ko-Fi link to help me make it better! 
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/O5O1FBP5F)

## Is it free?

Yes, 100% free-as-in-beer. It comes with no warranties, but you can do whatever you want with it. See license below.

## Can you help me deploy it / make a version that detects X / make a version for me? 

Sure, send me an email. I do lots of commercial perception stuff for UAV and other things. 

## How can I use this?

There are many ways you can use WALDO! 

### example 1: real-time inference!
Download the model weights and run with yolov7 straight from the command-line like this: \
python3 detect.py --weights best.pt --img-size 960 --save-txt --source /your_frames/ --project /your_save_folder/ 

### example 2: run on a single image!
Open the ONNX model from the /ONNX format with opencv and run your inference from there. 
Just take a look at the sample script run_local_onnx_boilerplate.py provided in the repo. 
If you have an Nvidia GPU set "cuda" in there to True, and point it at your files. If you're missing any dependencies you can use the provided requirements.txt to set them up... and that's all there is to it!

### example 3: run on a very very large image (like earth observation stuff):
Use the provided script called run_local_onnx_largeinput_tiled_process.py
Set "CUDA" to "True" if you have an Nvidia GPU, and point it to the image that you want to tile and process by changing this line:
img = cv2.imread('./Columbus_COWC_1.png')

The script will spit out a re-built full-size image (below is a crop): \
<img width="1202" alt="image" src="https://user-images.githubusercontent.com/20320678/233666879-34fc1101-4773-4f80-a99d-2a85d6189e1e.png">



....along with a count of objects to the console: \

<img width="378" alt="image" src="https://user-images.githubusercontent.com/20320678/233662619-caefb5c9-29bf-48b6-8f8b-78ddf1c59dda.png">


### example 4: export your own ONNX model with your favorite settings!
Load the .pt model and convert it to ONNX with your own settings using yolov7 (something like export.py --weights /best.pt  --grid --end2end --simplify --topk-all 100 --iou-thres 0.45 --conf-thres 0.3 --img-size 960 960 --max-wh 960 )  etc... read the export docs to figure out what you need.

Alternatively you can use the included ONNX model and deploy it *_wherever you want_* ! 


## Why? 

In the UAV space there is a need for FOSS AI tools that are usable by all for safety and ground risk mitigation! 
Additionally the space needs a reference system to serve as a benchmark for private-source comparables when defining security use cases with regulation authorities such as the FAA or EASA.


## What is next

Over the coming weeks I'm going to update this repository with variants of the main model optmized for specific use cases,<del> along with some boilerplate code for deployment! </del> DONE!

# Support this project :)
If you'd like to help me make this better please consider donating a few $$$ to keep my GPUs running using Ko-Fi:
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/O5O1FBP5F)


# Credits:
All code written by me and GPT4 😄

Thanks to https://gdo152.llnl.gov/cowc/ for the satellite image of Columbus, Ohio used in the tiling demo.


## Copyright is MIT license
Copyright Stephan Sturges 2023

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
