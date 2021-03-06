# Computer Vision On The Edge
This is the repo accompanying my Medium article "Edge AI - Computer vision on the edge", [Part 1](https://medium.com/datadriveninvestor/edge-ai-computer-vision-on-the-edge-dfa4ad604651), [Part 2](https://medium.com/datadriveninvestor/edge-ai-computer-vision-inference-on-the-edge-part-2-2-aaddfae870f0). 

Whenever network connectivity is not available, or network latency is insufficient, or there is a need for local processing due to regulatory, privacy and security concerns, or, indeed, a combination of all of these aspects is given, today’s standard cloud-based AI approach isn’t an option.

Edge AI, i.e., AI computations performed locally on the “edge” with little or no network connectivity using locally obtained data represents a viable alternative under these circumstances subject to the computation and data processing limitations of the edge device under consideration. 

In the Medium blog posts, I explain how to use the open source Intel OpenVINO library together with the Intel Neural Compute Stick 2 for edgde AI in the form of object detection on a commodity edge device, the Raspberry Pi.

The corresponding object detection demo application, which is produced in the course of the two blog posts, using [YOLOv3-tiny](https://pjreddie.com/darknet/yolo/) is shown below (input video source: CAMPUS dataset for multi-view object tracking - Y. Xu, X. Liu, L. Qin, and S.-C. Zhu, “Cross-view People Tracking by Scene-centered Spatio-temporal Parsing”, AAAI Conference on Artificial Intelligence (AAAI), 2017).

<p align="center">
  <img src="https://github.com/cm230/Computer-Vision-On-The-Edge/blob/master/demo.gif"/>
</p>

The __hardware setup__ looks as follows:
* Raspberry Pi 3 Model B+ (Raspberry Pi 4 should be just fine as well)
* Raspberry Pi on-board camera module
* Intel Neural Compute Stick 2

The __software__ side of things consists of:
* [Intel OpenVINO toolkit for Raspbian OS](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html) to be installed on the Raspberry Pi. The demos were generated with version: 2020.1.023
* Intel OpenVINO toolkit for ![Linux](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html), ![macOS](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_macos.html) or ![Windows](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html) for your main machine
* At least Python 3.5.6

I recommend to install [FFMPEG](https://ffmpeg.org/) on your Raspberry Pi to make it easier to find a working file extension and output video codec combination.

The custom-implementation of object detection applications using this hardware and software setup alongside the code in this repo is described in detail in [Part 2](https://medium.com/datadriveninvestor/edge-ai-computer-vision-inference-on-the-edge-part-2-2-aaddfae870f0) of my Medium article.
