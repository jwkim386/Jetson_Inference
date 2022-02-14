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

만약 [Docker container](aux-docker.md)도커 컨테이너를 사용하고 있다면, output(결과) 이미지를 마운드 된 `images/test` 디렉토리 아래 저장하는 것을 권장합니다. 그런 이미지들은 호스트 기기에서 `jetson-inference/data/images/test` 디렉토리 아래서 쉽게 볼 수 있습니다. 더 자세한 정보는 다음 페이지를 확인하세요. (for more info, see [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)).

아래는 기본 모델인 SSD-Mobilenet-v2 를 이용하여 보행자를 검출하는 예제입니다.

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

> **note**:  모델을 처음 실행시키면 network(모델)을 TensorRT가 모델을 optimize(최적화)하기 때문에 몇 분 정도가 소요될 수 있습니다.<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;해당 최적화된 network(모델)은 disk에 캐싱되며, 이후에는 금방 실행됩니다.

아래는 콘솔 프로그램들의 결과들입니다. SSD를 기반으로한 모델이 훈련된 [91-class](../data/networks/ssd_coco_labels.txt)개의 클래스를 갖는 MS COCO dataset은 사람, 차량, 동물과 여러 가정물품의 객체를 포함하고 있습니다. 

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-animals.jpg" >

다양한 이미지들이 `images/` 아래 테스트를 위해 제공됩니다. 이는 `cat_*.jpg`, `dog_*.jpg`, `horse_*.jpg`, `peds_*.jpg` 등등입니다.

### 디렉토리 혹은 연속된 이미지들의 Processing(처리)

만약 한번에 처리해야할 여러 이미지들이 있다면, `detectnet`을 수행할 때 디렉토리의 경로나 wildcard sequence(와일드카드 시퀀스)로 인자를 줄 수 있다.:

```bash
# C++
./detectnet "images/peds_*.jpg" images/test/peds_output_%i.jpg

# Python
./detectnet.py "images/peds_*.jpg" images/test/peds_output_%i.jpg
```

> **note:** 와일드카드를 사용할 때는 quotes(쌍따옴표)(`"*.jpg"`)로 잘 감쌀 수 있도록 합니다. 그리고, OS는 자동으로 해당 인자를 expand(확장)하여 커맨드 라인에 주어진 인자와 순서를 수정합니다. 이는 결과를 덮어쓸 수도 있으니 주의해야합니다.

이미지들을 loading/saving 하는 것에 대한 더 자세한 정보는 다음 페이지를 참고하세요.  [Camera Streaming and Multimedia](aux-streaming.md#sequences)

### 비디오 파일 Processing(처리)

disk에 저장된 비디오 파일로도 실습을 진행할 수 있습니다. `/usr/share/visionworks/sources/data` 아래에 테스트용 영상이 있습니다.

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

검출 sensitivity(민감도)를 조정하기 위해 `--threshold` 변경해볼 수 있음을 기억하세요.

### 사용 가능한 Pre-Trained(미리 훈련된) 객체 검출 모델 

아래 테이블은 [download](building-repo-2.md#downloading-models) 가능한 pre-trained(미리 훈련된) 모델들을 나타낸 것입니다. 또한 `--network` 플래그로 인자를 전달할 수 있습니다.

모델:
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

> **note**:  추가로 모델을 다운로드 받기 위해서는 [Model Downloader](building-repo-2.md#downloading-models) 를 실행하세요.<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>


### 다른 모델로 Running(수행하기)

위 테이블에서 CLI arguments에 대응하는 하나의 모델을 커맨드 라인에서의 `--network` 에 전달함으로써 어떤 모델을 사용할지 specify(선택)할 수 있습니다. By default(기본값으로), SSD-Mobilenet-v2 모델은 `--network` 플래그를 사용할 필요가 없습니다.

예로, 만약 SSD-Inception-v2 모델을 [Model Downloader](building-repo-2.md#downloading-models) 툴을 통해 다운도르 받았다면 아래와 같이 수행할 수 있습니다.:

``` bash
# C++
$ ./detectnet --network=ssd-inception-v2 input.jpg output.jpg

# Python
$ ./detectnet.py --network=ssd-inception-v2 input.jpg output.jpg
```

### 소스 코드

참고로, 아래 코드는 [`detectnet.py`](../python/examples/detectnet.py)의 소스코드입니다.:

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

다음으로, 라이브 카메라 스트림을 통해 객체 검출을 진행하도록 하겠습니다.

##
<p align="right">Next | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></b>
<br/>
Back | <b><a href="imagenet-camera-2.md">Running the Live Camera Recognition Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
