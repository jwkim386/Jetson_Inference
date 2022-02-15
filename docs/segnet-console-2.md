<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-example-2.md">Back</a> | <a href="segnet-camera-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# SegNet 으로 하는 Semantic Segmentation(의미론적 분할..)
해당 튜토리얼에서 다룰 다음 딥러닝 기술은 **semantic segmentation(의미론적 분할)**입니다. Semantic segmentation(의미론적 분할)은 이미지 인식 기술에 기반합니다. 단지 전체 이미지가 아닌 픽셀 단위로 이루어지는 점이 앞선 이미지 인식들과는 조금 다릅니다. 이를 *convolutionalizing(컨볼루셔널라이징)*이라 하며, 이는 pre-trained(훈련된) 모델로 픽셀별로 라벨링할 수 있도록 [Fully Convolutional Network (FCN)](https://arxiv.org/abs/1605.06211) 으로 transform(변환)한 것입니다. 특히 배경을 인식함에 있어서 효과적인데, segmentation(분할)은 조밀한 픽셀 단위로 background(배경)을 포함한 이미지의 다양한 객체들을 분리합니다.

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation.jpg">

[`segNet`](../c/segNet.h) 은 input(입력)으로 2D 이미지를 받고 같은 크기의 픽셀별로 분류된 overlay(오버랩핑)된 이미지를 output(출력)합니다. 각 mask(마스크)의 픽셀들은 분류된 객체에 대응합니다.   [`segNet`](../c/segNet.h) 은 [Python](https://rawgit.com/dusty-nv/jetson-inference/pytorch/docs/html/python/jetson.inference.html#segNet) and [C++](../c/segNet.h) 으로 사용할 수 있습니다.

`segNet`의 예제로 아래 C++, Python 코드가 제공됩니다.

- [`segnet.cpp`](../examples/segnet/segnet.cpp) (C++) 
- [`segnet.py`](../python/examples/segnet.py) (Python) 

위 예제들은 이미지, 비디오, 카메라 스트림을 segmentation(분할)할 수 있습니다. input/output 스트림에 대한 자세한 정보는 다음 페이지를 참고하세요. [Camera Streaming and Multimedia](aux-streaming.md) page.

[pre-trained models available](#pretrained-segmentation-models-available) 을 살펴보면 Jetson에서 realtime performance(실시간으로 결과를 낼만큼 빠른 성능)을 가지고 FCN-ResNet18을 backbone(백본)으로 하는 더 다향한 segmentation(분할) 모델이 준비돼있습니다. 도시, 비포장 도로, 사무실 내부와 같은 다양한 환경과 물체들을 segmentation(분할)할 수 있는 다양한 모델을 제공합니다.

### 사용 가능한 Pre-Trained(훈련된) Segmentation(분할)

아래 테이블은 [download](building-repo-2.md#downloading-models) 가능한 pre-trained(훈련된) 모델들이며, `--network` 플래그를 사용할 때 참고해야합니다. 아래 모델들은 21-class FCN-ResNet18 모델을 기반으로 하고 [PyTorch](https://github.com/dusty-nv/pytorch-segmentation) 를 이용하여 다양한 데이터셋으로 훈련했습니다. 그리고 .TensorRT로 불러오기 위해 [ONNX format](https://onnx.ai/)로 export(변환) 하였습니다.

| Dataset      | Resolution | CLI Argument | Accuracy | Jetson Nano | Jetson Xavier |
|:------------:|:----------:|--------------|:--------:|:-----------:|:-------------:|
| [Cityscapes](https://www.cityscapes-dataset.com/) | 512x256 | `fcn-resnet18-cityscapes-512x256` | 83.3% | 48 FPS | 480 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 1024x512 | `fcn-resnet18-cityscapes-1024x512` | 87.3% | 12 FPS | 175 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 2048x1024 | `fcn-resnet18-cityscapes-2048x1024` | 89.6% | 3 FPS | 47 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 576x320 | `fcn-resnet18-deepscene-576x320` | 96.4% | 26 FPS | 360 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 864x480 | `fcn-resnet18-deepscene-864x480` | 96.9% | 14 FPS | 190 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 512x320 | `fcn-resnet18-mhp-512x320` | 86.5% | 34 FPS | 370 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 640x360 | `fcn-resnet18-mhp-640x360` | 87.1% | 23 FPS | 325 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 320x320 | `fcn-resnet18-voc-320x320` | 85.9% | 45 FPS | 508 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 512x320 | `fcn-resnet18-voc-512x320` | 88.5% | 34 FPS | 375 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 512x400 | `fcn-resnet18-sun-512x400` | 64.3% | 28 FPS | 340 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 640x512 | `fcn-resnet18-sun-640x512` | 65.1% | 17 FPS | 224 FPS |

* 만약 CLI argument에서 resolution(해상도)가 생략되면 가장 낮은 resolution(해상도)를 갖는 모델이 로드됩니다. 
* Accuracy(정확도)는 validation set(검증셋) 에 대한 픽셀별 classification(분류)의 정확도를 의미합니다.
* Performance(성능)은 GPU FP16 mode와 JetPack 4.2.1, `nvpmodel 0` (MAX-N) 에서 측정되었습니다.

> **note**: 추가적인 모델을 다운로드하고 싶다면, 다음을 실행하세요 [Model Downloader](building-repo-2.md#downloading-models)<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>

### 커맨드 라인에서 이미지 Segmenting(분할)하기

먼저, 이미지를 분할해봅시다. input/output 경로 옵션외에, 몇몇 옵션들이 더 있는데 아래와 같습니다.

- optional `--network` 플래그는 사용하고자 하는 모델을 바꿉니다. (see [above](#pre-trained-segmentation-models-available))
- optional `--visualize` 플래그는 `mask` 나 `overlay` 혹은 둘 다 가능합니다. (default(기본값) 은 `overlay`)
- optional `--alpha` `overlay`의 알파값을 설정합니다. (default(기본값)은 `120`)
- optional `--filter-mode` 플래그는 `point` 나 `linear` 를 사용할 수 있습니다. (default(기본값)은 `linear`)

더 많은 정보를 원하시면 어플리케이션을 실행할 때 `--help` 플래그를 사용하시고, input/output 프로토콜에 대한 자세한 정보는 다음 페이지를 참고하세요.[Camera Streaming and Multimedia](aux-streaming.md)

아래는 프로그램의 실행 예제입니다.:

#### C++

``` bash
$ ./segnet --network=<model> input.jpg output.jpg                  # overlay segmentation on original
$ ./segnet --network=<model> --alpha=200 input.jpg output.jpg      # make the overlay less opaque
$ ./segnet --network=<model> --visualize=mask input.jpg output.jpg # output the solid segmentation mask
```

#### Python

``` bash
$ ./segnet.py --network=<model> input.jpg output.jpg                  # overlay segmentation on original
$ ./segnet.py --network=<model> --alpha=200 input.jpg output.jpg      # make the overlay less opaque
$ ./segnet.py --network=<model> --visualize=mask input.jpg output.jpg # output the segmentation mask
```

### Cityscapes

다른 시나리오를 살펴봅시다. 아래는 [Cityscapes](https://www.cityscapes-dataset.com/) 모델을 사용한 도시 도로의 segmenting(분할) 예제입니다.

``` bash
# C++
$ ./segnet --network=fcn-resnet18-cityscapes images/city_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-cityscapes images/city_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-city.jpg" width="1000">

Cityscapes 모델에 적용해볼 다른 `city-*.jpg`와 같은 테스트 이미지들이 `images/` 디렉토리 아래에 더 있습니다.

### DeepScene

The [DeepScene dataset](http://deepscene.cs.uni-freiburg.de/) 은 오프로드의 길과 식생들로 이루어져있으며, 야외에서 동작하는 로봇의 path-following(길찾기)에 도움이 됩니다.
아래는 `--visualize` 플래그로 segmentation(분할) overlay와 mask를 생성한 예제입니다. 

#### C++
``` bash
$ ./segnet --network=fcn-resnet18-deepscene images/trail_0.jpg images/test/output_overlay.jpg                # overlay
$ ./segnet --network=fcn-resnet18-deepscene --visualize=mask images/trail_0.jpg images/test/output_mask.jpg  # mask
```

#### Python
``` bash
$ ./segnet.py --network=fcn-resnet18-deepscene images/trail_0.jpg images/test/output_overlay.jpg               # overlay
$ ./segnet.py --network=fcn-resnet18-deepscene --visualize=mask images/trail_0.jpg images/test/output_mask.jpg # mask
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-deepscene-0-overlay.jpg" width="850">
<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-deepscene-0-mask.jpg">

There are more sample images called `trail-*.jpg` located under the `images/` subdirectory.

### Multi-Human Parsing (MHP)

[Multi-Human Parsing](https://lv-mhp.github.io/) 데이터셋은 다양한 종류의 옷, 팔, 다리, 머리등과 같은 신체 일부분을 조밀하게 라벨링한 데이터셋입니다.
`images/`아래에 있는 `humans-*.jpg` 테스트 이미지를 MHP 모델에 적용해봅시다.

``` bash
# C++
$ ./segnet --network=fcn-resnet18-mhp images/humans_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-mhp images/humans_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-mhp-0.jpg" width="825">
<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-mhp-1.jpg" width="825">

#### MHP Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-mhp-legend.jpg">

### Pascal VOC

[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 은 근본 데이터셋 중 하나이며, 다양한 사람, 동물, 차량, 가정물품등을 포함합니다. `object-*.jpg` 와 같은 이름으로 샘플 이미지들이 있습니ㅏ다.

``` bash
# C++
$ ./segnet --network=fcn-resnet18-voc images/object_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-voc images/object_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-voc.jpg" width="1000">

#### VOC Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-voc-legend.jpg">

### SUN RGB-D

The [SUN RGB-D](http://rgbd.cs.princeton.edu/) 데이터셋은 사무실이나 집안 환경에서 쉽게 찾아볼 수 있는 물체에 대한 segmentation(분할) groun-truth(정답)를 제공합니다. `images/` 아래에  `room-*.jpg` 이미지들을 찾아서 SUN 모델에 적용해봅시다.:

``` bash
# C++
$ ./segnet --network=fcn-resnet18-sun images/room_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-sun images/room_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-sun.jpg" width="1000">

#### SUN Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-sun-legend.jpg">

### 디렉토리, 혹은 한 번에 많은 이미지 Processing(처리하기)

한 번에 많은 이미지들을 처리하려면 경로에 인자에 디렉토리 경로를 주거나 wildcard sequence(와일드 카드)를 사용할 수 있습니다. 

``` bash
# C++
$ ./segnet --network=fcn-resnet18-sun "images/room_*.jpg" images/test/room_output_%i.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-sun "images/room_*.jpg" images/test/room_output_%i.jpg
```

> **note:** wildcards(와일드카드)를 사용할 때는, 항상 quotes(쌍따옴표)로 잘 감싸야합니다. (`"*.jpg"`). 그렇지 않으면 OS가 자동으로 커맨드 라인의 인수 순서를 수정하고 이는 input(입력) 이미지 중 하나가 output(출력)에 의해 덮어쓰여질 수 있습니다. 

이미지 loading/saving에 대한 더 자세한 정보는 다음 페이지를 확인하세요.  [Camera Streaming and Multimedia](aux-streaming.md#sequences)

##
<p align="right">Next | <b><a href="segnet-camera-2.md">Running the Live Camera Segmentation Demo</a></b>
<br/>
Back | <b><a href="detectnet-example-2.md">Coding Your Own Object Detection Program</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
