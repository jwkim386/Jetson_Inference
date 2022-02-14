<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-example-2.md">Back</a> | <a href="detectnet-console-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# 라이브 카메라로 이미지 인식 데모 실행시키기

이전에 사용되었던 [`imagenet.cpp`](../examples/imagenet/imagenet.cpp) / [`imagenet.py`](../python/examples/imagenet.py) 예제들은 실시간 카메라 스트리밍에도 이용할 수 있다. 다음은 지원되는 카메라의 종류를 나열한 것이다.:

- MIPI CSI cameras (`csi://0`)
- V4L2 cameras (`/dev/video0`)
- RTP/RTSP streams (`rtsp://username:password@ip:port`)

비디오 스트리밍에 대한 자세한 정보는 다음을 확인할 수 있습니다.[Camera Streaming and Multimedia](aux-streaming.md) page.

아래는 카메라로 수행하는 전형적인 예제가 있습니다.

#### C++

``` bash
$ ./imagenet csi://0                    # MIPI CSI camera
$ ./imagenet /dev/video0                # V4L2 camera
$ ./imagenet /dev/video0 output.mp4     # save to video file
```

#### Python

``` bash
$ ./imagenet.py csi://0                 # MIPI CSI camera
$ ./imagenet.py /dev/video0             # V4L2 camera
$ ./imagenet.py /dev/video0 output.mp4  # save to video file
```

> **note**: 예제 카메라들을 사용하기 위해 아래 Jetson Wiki의 섹션들을 확인하세요.: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

OpenGL 윈도우에서 디스플레이 되는 것은 라이브 카메라 스트리밍, 분류된 이미지의 이름, 그리고 분류된 객체일 확률입니다. 그리고 network(모델)의 주사율입니다. 젯슨 나노에서는 GoogleNet, ResNet18로 ~75FPS 정도까지 볼 수 있습니다.

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/imagenet_camera_bear.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/imagenet_camera_camel.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/imagenet_camera_triceratops.jpg" width="800">

1000개의 클래스로 구분돼있던 ImageNet으로 훈련된 모델을 사용했기 때문에 해당 어플리케이션도 1000개의 이미지에 대해 분류할 수 있습니다. 1000개의 맵핑된 객체들의 타입은 다음 repo에서 찾아볼 수 있습니다.[`data/networks/ilsvrc12_synset_words.txt`](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)

이것으로 Hello AI World의 이미지 분류 튜토리얼을 마치겠습니다. 다음으로는 object detection(객체 검출)을 진행하도록 하겠습니다. 해당 모델은 매 프레임마다 객체들의 boundinb box를 output(출력)으로 합니다.

##
<p align="right">Next | <b><a href="detectnet-console-2.md">Locating Object Coordinates with DetectNet</a></b>
<br/>
Back | <b><a href="imagenet-example-2.md">Coding Your Own Image Recognition Program</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
