W.A.L.D.O.
Whereabouts Ascertainment for Low-lying Detectable Objects 
---------------------------------------------------------------------


## /!\ Look out: the latest versions of the model are on HF [here](https://huggingface.co/StephanST/WALDO30)

[![WALDO 3.0 preview vid](https://i.imgur.com/hGghrLn.jpeg)](https://www.youtube.com/watch?v=1y5y9yklj2U)

Welcome to the WALDO v3.0 public release
---------------------------------------------------------------------


WHAT IS WALDO?

WALDO is a detection AI model, based on a large YOLO-v8 backbone and my own
synthetic data pipeline. **The model is capable of detecting these classes 
of items in overhead imagery ranging in altitude from about 30 feet to 
satellite imagery!**


Output classes:

0 -> 'LightVehicle'  --> all kinds of civilan cars, including pickup trucks, vans etc... 🚗🏎️🚓🚐🚑 </br>
1 -> 'Person' --> people! all kinds of people including ones that are on bikes or swimming in the sea 🧍‍♀️🕺💃🧜🏽‍♀️🏂🧞</br> 
2 -> 'Building' --> all kinds of buildings 🕌🏛️🏭🏡</br>
3 -> 'UPole' --> utility poles, power poles, anything thin and sticking up that you should avoid with a drone 🎏</br>
4 -> 'Boat' --> boats, ships, canoes, kayaks, surf boards... all the floaty stuff 🚢🏄</br>
5 -> 'Bike' --> bikes, mopeds, motorbikes, all stuff with 2 wheels 🚲</br>
6 -> 'Container' --> shipping containers, including on the back of an articulated truck... 📦🏗️</br>
7 -> 'Truck' --> large commercial vehicles including articulated trucks or big box-on-chassis delivery trucks 🚚</br>
8 -> 'Gastank'--> cylindrical tanks such as butane tanks and gas expansion tanks, or grain silos... pretty much anything that looks cylindrical for storing liquids 🫙</br>
10 -> 'Digger' --> all kinds of construction vehicles, including tractors and construction gear 🚜</br>
11 -> 'Solarpanels' --> solar panels ▪️🌞▪️</br>
12 -> 'Bus' --> a bus 🚌</br>

--> In general the lower the class number the better-trained you can expect it to be.
For users of previous versions of WALDO: note that I removed the military class and smoke detection. This is meant to be a FOSS tool for civilian use and I don't want to pursue making it work for military applications.


---------------------------------------------------------------------

WHERE IS WALDO?

Right here on HF -> https://huggingface.co/StephanST/WALDO30

Note there are a couple more models that have slightly better performance over on Gumroad here: https://6228189440665.gumroad.com/l/WALDOv3
Those are for sale as a kind of sponsorship for the project: if you find value in the free ones here you can buy those for a nice little performance boost... but it's entirey up to you! 


[![P2 model performance boost](https://i.imgur.com/VKa5NN5.png)]


In both cases the actual files are MIT license and you can freely share them, so if someone gives you the ones from Gumroad you are free yo use them including commercially. It's really just a way to offset some of the work and compute that went into making this project and keeping it FOSS.


---------------------------------------------------------------------                                                                                                                                                               

WHAT IS IT GOOD FOR?

People are currently using versions of WALDO for:
1. disaster recovery
2. monitoring wildlife sanctuaries (intruder detection)
3. occupancy calculation (parking lots etc..)
4. monitoring infrastructure 
5. construction site monitoring
6. traffic flow management
7. crowd counting
8. some fun AI art applications!
9. drone safety (avoiding people / cars on the ground)
10. lots of other fun stuff...

The main reason for me to make WALDO free has in fact been discovering all these cool applications. Let me know what you build!

---------------------------------------------------------------------                                                                                                                                                               

FOR AI NERDS !

It's a set of YOLOv8 model, trained on my own datasets of synthetic and "augmented" / semi-synthetic data.
I'm not going to release the dataset for the time being.

The weights are completely open, allowing you to deploy in any number of ways this time! 


---------------------------------------------------------------------                                                                                

HOW CAN I START WITH WALDO?  

Check out the boilerplate code in the repo to run the models and output pretty detections using the wonderful Supervision annotation library from Roboflow :) 

---------------------------------------------------------------------

GOING DEEPER

Of course if you know your way around deploying AI models there is a lot more you do
with this release, inclusing:

1. fine-tuning the models on your own data (if you know what you are doing, this is probably your starting point)
2. building a nicely optimized sliding-window inference setup that works nicely on your edge hardware
3. quantizing the models for super-duper edge performance on cheap devices
4. using the models to annotate your own data and train something of your own!


Enjoy!

---------------------------------------------------------------------


PREVIOUS VERSIONS

I am retiring the old versions, this is the only one that will stay online.

---------------------------------------------------------------------


CAN YOU HELP ME WITH X? 

Sure, email me at stephan.sturges@gmail.com


---------------------------------------------------------------------


DETECTION OF X ISN'T WORKING AS EXPECTED:

I'd love to see example images, videos, sample data, etc at:
stephan.sturges@gmail.com


---------------------------------------------------------------------

SUPPORT WALDO!

Visit [![the WALDO gumroad page](https://t.co/kRvhYkVxW2)] to support the project!

---------------------------------------------------------------------


LICENSE
----------------------------------------------------------------------------

Unless otherwise specified all code in this release is published with the 
licence conditions below.
----------------------------------------------------------------------------


MIT License

Copyright (c) 2024 Stephan Sturges / Aircortex.com 

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
