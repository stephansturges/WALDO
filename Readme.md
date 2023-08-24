W.A.L.D.O.
Whereabouts Ascertainment for Low-lying Detectable Objects !

---------------------------------------------------------------------

Welcome to the WALDO v2.5 FINAL release! 🥳🥳🥳🥳

![fe361d16-c588-47ad-bff4-1c5185c0cd9f](https://github.com/stephansturges/WALDO/assets/20320678/3a5ad37c-db34-4d71-8a88-325672427b7a)

---------------------------------------------------------------------

Thanks to all participants in the beta! I had over 3000 sign-ups for the 
beta release and iterated really fast... I hope you'll like the result!

I am assuming you have  some experience with deployment of AI systems, 
but if you have any trouble using this release you can contact me at 
stephan.sturges at gmail

---------------------------------------------------------------------


WHAT IS WALDO?

WALDO is a detection AI model, based on a large YOLO-v7 backbone and my own
synthetic data pipeline. The basic model shared here, which is the only 
one published as FOSS at the moment, is capable of detecting these classes 
of items in overhead images ranging in altitude from about 30 feet to 
satellite imagery with a resolution of 50cm per pixel or better.


Well trained classes:
1. 'car'  --> all kinds of civilan cars, including pickup trucks
2. 'van' --> all kinds of civilian vans, gets confused with "car" a lot. You might want to fuse them! 🚗
3. 'truck' --> all kinds of box-trucks, flatbeds or articulated trucks, NOT small pickup trucks 🚚 
4. 'building' --> buildings of all kinds 🏣 
5. 'human' --> people! 🧍
6. 'gastank'--> cylindrical tanks such as butane tanks and gas expansion tanks, or grain silos 🫙
7. 'digger' --> all kinds of construction vehicles, including tractors and construction gear 🚜
8. 'container' --> shipping containers, including on the back of an articulated truck
9. 'bus' --> a bus 🚌
10. 'u_pole' --> utility poles, power poles, anything thin and sticking up that you should avoid with a plane 🎏
11. 'boat' --> boats 🚢
12. 'bike' --> bikes, mopeds, motorbikes, all things with 2 wheels 🚲 
13. 'smoke' --> smoke and fire 🔥🔥🔥
14. 'solarpanels' --> solar panels
15. 'arm/mil' --> this class detects certain types of armored vehicles (very unreliable for now, don't use it yet)
16. 'plane' --> planes (very unreliable for now, probably not worth using yet)


---------------------------------------------------------------------

WHARE IS WALDO?

Due to the size of the model files and the constraints of github LFS the files
are no longer stored directly on Github, please download the latest package 
using the link below:

https://bit.ly/3P7UdZ6


---------------------------------------------------------------------                                                                                                                                                               

FOR AI NERDS !

It's a big set of YOLOv7 model, trained on my own datasets of synthetic and "augmented" / semi-synthetic data.
I'm not going to release the dataset for the time being.

The ONNX models are exported for onnx-runtime with a batch-size of 1 and a max input size corresponding to the 
the network dimensions. They are also set up to export only the top 200 highest-confidence objects in most cases.

I'm planning to set up a way for people to get the .pt files and the ONNX models with unlimited outputs
for people who support further development of the project on Ko-Fi (https://ko-fi.com/stephansturges), the goal
being to offset some of the cost of training these networks (over 60K USD spent on AWS to date! 😅) 


---------------------------------------------------------------------                                                                                

HOW CAN I START WITH WALDO?  

Setup the environment with python3:
1. (optional) create a virtual python env for the project
2. install dependencies using the requirements file: pip install -r requirements.txt 

You may need to install a couple of other bits and pieces depending on your python3 env...
If you find anything really blocking send me an email and I'll update this readme.


---------------------------------------------------------------------  

RUN THE MODELS USING THE BOILERPLATE CODE IN /playground:

1. To run on video: 
put one or multiple .mp4 files in the ./input_vids subfolder, and then run:

python3 run_local_onnx_on_videos.py 

This will run the detection network in default settings and save an annotated video to
the ./output_vids/ subfolder.

You can also use the following command-line arguments:

python3 run_local_onnx_on_videos.py --frame_limit 3000 --frame_skip 8

"frame limit" defines where to stop processing the video, if you only want to test it
on the first 1000 frame then use --frame_limit 1000 for example

"frame skip" allows you to skip frames to keep processing quicker for testing, so
if your video is 30 fps and you only one 1 frame per second to be AI-annotated
then you can use --frame_skip 30 for instance


2. To run on a single image of any size:

Put some images in ./images_in/ and run:

python3 run_local_onnx_on_images.py

This will run detection on all images in the input folder and save the annotated
output images in the output folder, along with the txt files of the detections
in YOLO format. 

If the image is LARGER than 960x960px format it will be tiled into squares of 960px with
a litte overlap for analysis and then merged back together, so you can process
huge satellite images for example without needing to split them first.

If you want to run the network on a single image that should be processed at native resolution
you can use the "--resize" flag like this:

python3 run_local_onnx_on_images.py --resize

The output can be found in ./images_out/, you'll get images with pretty overlays and .txt files
with the actual detections


---------------------------------------------------------------------



WHAT IS INCLUDED? 

In the FOSS package there are a bunch of networks in ONNX format prepared for ONNXruntime, as
well as a few examples of networks in other export formats. Only the "V7-base/square/416px" 
network is included in all formats as part of this release, meaning you get a selection of
ONNX exported models including some quantized and prepared for Nvidia TensorRT, and you
also have the raw .pt files for the training run so that you can export your own.
I also added the base .pt files for the 512px V7 model.
These files also exist for each other network (or can be exported), but I'm thinking about
how to make those available for people who support the future development of WALDO in order
to support the cost of AI model training (which is over 50K $ already up to this point).
Reach out to me via email if you want a model / export that isn't in here!


/!\ Some tips for use:
- In real-world use cases you may want to merge classes 1 & 2 since there this still 
a lot of confusion between those classes
- The models are exported with non-maximum-suppression, so if you are using the
AI system in cases where objects are occluded by one another you will only get
the "most valid" object in most cases. 


Some of the network that is in this repo is very large, and is meant to be run on 
an inference server, and some are made for embedding on tiny edge devices... take 
a look around and find one that works for you!


---------------------------------------------------------------------

GOING DEEPER

Of course if you know your way around deploying AI models there is a lot more you do
with this release, inclusing:

1. There are certain models already released in CoreML format for iOS, give those a try
2. There are some models that are exported for TensorRT, including some cool quantization!
3. For a couple of models the .pt files are included in this release, play with making
your own exports or running thos directly using YOLOv7 from https://github.com/WongKinYiu/yolov7
4. Get yourself a cool, cheap, little AI camera from Luxonis and run one of the OpenVino blobs
that are currently exported for the V7-base/416px network and the V7-tiny/512px network. These
are super cool and do excellent AI detections directly on 15g hardware that costs <200$... crazy stuff.
5. Build your own commercial application!


Enjoy!

---------------------------------------------------------------------



PREVIOUS VERSIONS

You can find the repo with WALDO v1.0 here:
https://github.com/stephansturges/WALDO


---------------------------------------------------------------------


CAN YOU HELP ME WITH X? 

Sure, email me at stephan.sturges@gmail.com


---------------------------------------------------------------------


DETECTION OF X ISN'T WORKING AS EXPECTED:

I'd love to see example images, videos, sample data, etc at:
stephan.sturges@gmail.com


HOW DOES AIRCORTEX MAKE MONEY?

Aircortex' mission statement is to make the SOTA in ground-risk AI and sensing,
and to make the basic models free and easy to use for both hobbyists and 
professionals in the UAV / AAM industry, to acclerate safe access to the skies
in the 21st century.


Aircortex is an "open-core" AI company: the basic model is completely
free and open-source for anyone to use including in commercial products.

I make money by charging for: 
1. help with training additional detection classes, 
2. retraining for your specific hardware, 
3. building the software stack to support specific deployment cases, 
4. helping companies set up the right hardware architecture for AI integration,
5. custom hardware setups for specific environments
6. more "feature-complete" versions of my FOSS products such as integrating 3D perception
etc... 

Contact me at stephan.sturges@gmail.com to find out more.

---------------------------------------------------------------------

SUPPORT WALDO!

Training this base model took about 3 months of work and ~20K$ in cloud compute.
If you find value in it, please support development of the next version on:
https://ko-fi.com/stephansturges

You can also sign-up there to be a sponsor of WALDO for 500$ / month and get 
early access to future models.

 ____    ____    ____    ____    ____    ____    ____    ____    ____    ____   
/\____/\/\____/\/\____/\/\____/\/\____/\/\____/\/\____/\/\____/\/\____/\/\___ 
\/____\/\/____\/\/____\/\/____\/\/____\/\/____\/\/____\/\/____\/\/____\/\/___


LICENSE
----------------------------------------------------------------------------

Unless otherwise specified all code in this release is published with the 
licence conditions below.
----------------------------------------------------------------------------


MIT License

Copyright (c) 2023 Stephan Sturges / Aircortex.com 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


