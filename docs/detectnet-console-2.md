<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-camera-2.md">Back</a> | <a href="detectnet-camera-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></s></p>

# DetectNet 으로 객체의 위치 찾기

이전의 recognition(인식) 예제에서는 전체 이미지에 대한 객체의 클래스 확률을 구했습니다. 지금부터는 **object detection(객체 검출)** 에 집중하고자 합니다. 그리고 매 프레임에 객체가 어디있는지 bounding box를 구함으로써 알아보겠습니다. 이미지 분류와는 다르게 객체 검출에서는 하나의 프레임에서 많은 객체를 구분해낼 수 있습니다.

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet.jpg" >

The [`detectNet`](../c/detectNet.h) 은 이미지를 input(입력)으로 받고 output(출력)은 검출된 객체들의 bounding box의 좌표들과 이에 대응하는 확률값입니다. [`detectNet`](../c/detectNet.h) 은 [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#detectNet) 혹은 [C++](../c/detectNet.h) 로 사용 가능합니다.  아래 다양한 모델들을 살펴볼 수 있습니다. [pre-trained detection models](#pre-trained-detection-models-available) 들을 다운로드 받을 수 있습니다. default(기본) 모델은 [91-class](../data/networks/ssd_coco_labels.txt) 개의 클래스를 검출할 수 있는 SSD-Mobilenet-v2 모델입니다. 이는 MS COCO dataset으로 훈련되었고 TensorRT로 실시간처리 성능을 만족하였습니다. 

`detectNet`의 사용 예제로써 C++, Python 샘플 프로그램을 제공합니다.:

- [`detectnet.cpp`](../examples/detectnet/detectnet.cpp) (C++) 
- [`detectnet.py`](../python/examples/detectnet.py) (Python) 

위 샘플들은 이미지, 비디오, 카메라 스트리밍에 대해 사용 가능합니다. 다양한 타입의 input/output 스트림 지원에 대한 정보를 얻고 싶으면 다음 페이지를 방문하세요. [Camera Streaming and Multimedia](aux-streaming.md)

### 이미지에서 객체 검출하기

먼저 `detectnet` 프로그램을 이미지 한 장에 대해서 적용해봅시다. input/output 경로 외에도 몇몇 커맨드 라인 옵션들이 있습니다.

옵션:
- optional `--network` 플래그는 사용할 [detection model](detectnet-console-2.md#pre-trained-detection-models-available) 을 바꿉니다.
- optional `--overlay` 플래그는 컴마로 구분하는 `box`, `labels`, `conf`나 `none`과 같은 조합들로 주어질 수 있습니다. default(기본값)은 `--overlay=box,labels,conf` 이며 box, label,(confidence)확률값을 출력합니다.	
- optional `--alpha` 값은 overlay 할 때 알파값을 결정합니다. (the default(기본값)은 `120`).	
	
- optional `--threshold` 최소 threshold 값을  결정한다. (the default(기본값)은 `0.5`).

If you're using the [Docker container](aux-docker.md), it's recommended to save the output images to the `images/test` mounted directory.  These images will then be easily viewable from your host device under `jetson-inference/data/images/test` (for more info, see [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)). 

Here are some examples of detecting pedestrians in images with the default SSD-Mobilenet-v2 model:

``` bash
# C++
$ ./detectnet --network=ssd-mobilenet-v2 images/peds_0.jpg images/test/output.jpg     # --network flag is optional

# Python
$ ./detectnet.py --network=ssd-mobilenet-v2 images/peds_0.jpg images/test/output.jpg  # --network flag is optional
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-peds-0.jpg" >

``` bash
# C++
$ ./detectnet images/peds_1.jpg images/test/output.jpg

# Python
$ ./detectnet.py images/peds_1.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-peds-1.jpg" >

> **note**:  the first time you run each model, TensorRT will take a few minutes to optimize the network. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;this optimized network file is then cached to disk, so future runs using the model will load faster.

Below are more detection examples output from the console programs.  The [91-class](../data/networks/ssd_coco_labels.txt) MS COCO dataset that the SSD-based models were trained on include people, vehicles, animals, and assorted types of household objects to detect.

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-animals.jpg" >

Various images are found under `images/` for testing, such as `cat_*.jpg`, `dog_*.jpg`, `horse_*.jpg`, `peds_*.jpg`, ect. 

### Processing a Directory or Sequence of Images

If you have multiple images that you'd like to process at one time, you can launch the `detectnet` program with the path to a directory that contains images or a wildcard sequence:

```bash
# C++
./detectnet "images/peds_*.jpg" images/test/peds_output_%i.jpg

# Python
./detectnet.py "images/peds_*.jpg" images/test/peds_output_%i.jpg
```

> **note:** when using wildcards, always enclose it in quotes (`"*.jpg"`). Otherwise, the OS will auto-expand the sequence and modify the order of arguments on the command-line, which may result in one of the input images being overwritten by the output.

For more info about loading/saving sequences of images, see the [Camera Streaming and Multimedia](aux-streaming.md#sequences) page.

### Processing Video Files

You can also process videos from disk.  There are some test videos found on your Jetson under `/usr/share/visionworks/sources/data`

``` bash
# C++
./detectnet /usr/share/visionworks/sources/data/pedestrians.mp4 images/test/pedestrians_ssd.mp4

# Python
./detectnet.py /usr/share/visionworks/sources/data/pedestrians.mp4 images/test/pedestrians_ssd.mp4
```

<a href="https://www.youtube.com/watch?v=EbTyTJS9jOQ" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-pedestrians-video.jpg width="750"></a>

``` bash
# C++
./detectnet /usr/share/visionworks/sources/data/parking.avi images/test/parking_ssd.avi

# Python
./detectnet.py /usr/share/visionworks/sources/data/parking.avi images/test/parking_ssd.avi
```

<a href="https://www.youtube.com/watch?v=iB86W-kloPE" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-parking-video.jpg width="585"></a>

Remember that you can use the `--threshold` setting to change the detection sensitivity up or down (the default is 0.5).

### Pre-trained Detection Models Available

Below is a table of the pre-trained object detection networks available for [download](building-repo-2.md#downloading-models), and the associated `--network` argument to `detectnet` used for loading the pre-trained models:

| Model                   | CLI argument       | NetworkType enum   | Object classes       |
| ------------------------|--------------------|--------------------|----------------------|
| SSD-Mobilenet-v1        | `ssd-mobilenet-v1` | `SSD_MOBILENET_V1` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| SSD-Mobilenet-v2        | `ssd-mobilenet-v2` | `SSD_MOBILENET_V2` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| SSD-Inception-v2        | `ssd-inception-v2` | `SSD_INCEPTION_V2` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| DetectNet-COCO-Dog      | `coco-dog`         | `COCO_DOG`         | dogs                 |
| DetectNet-COCO-Bottle   | `coco-bottle`      | `COCO_BOTTLE`      | bottles              |
| DetectNet-COCO-Chair    | `coco-chair`       | `COCO_CHAIR`       | chairs               |
| DetectNet-COCO-Airplane | `coco-airplane`    | `COCO_AIRPLANE`    | airplanes            |
| ped-100                 | `pednet`           | `PEDNET`           | pedestrians          |
| multiped-500            | `multiped`         | `PEDNET_MULTI`     | pedestrians, luggage |
| facenet-120             | `facenet`          | `FACENET`          | faces                |

> **note**:  to download additional networks, run the [Model Downloader](building-repo-2.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>


### Running Different Detection Models

You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, SSD-Mobilenet-v2 if the optional `--network` flag isn't specified.

For example, if you chose to download SSD-Inception-v2 with the [Model Downloader](building-repo-2.md#downloading-models) tool, you can use it like so:

``` bash
# C++
$ ./detectnet --network=ssd-inception-v2 input.jpg output.jpg

# Python
$ ./detectnet.py --network=ssd-inception-v2 input.jpg output.jpg
```

### Source Code

For reference, below is the source code to [`detectnet.py`](../python/examples/detectnet.py):

``` python
import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)

	# render the image
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
```

Next, we'll run object detection on a live camera stream.

##
<p align="right">Next | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></b>
<br/>
Back | <b><a href="imagenet-camera-2.md">Running the Live Camera Recognition Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
