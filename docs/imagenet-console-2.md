* Auth : 김준호
* Data : 2022-02-14 작성

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="building-repo-2.md">Back</a> | <a href="imagenet-example-python-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# Classifying Images with ImageNet으로 이미지 분류하기

사용가능한 여러 종류의 딥러닝 (networks)모델들이 있습니다. 가령, recognition(인식), detection/localization(검출/위치 찾기) 그리고 semantic segmentation이 있습니다. 처음으로 사용해볼 딥러닝 기술은 바로 위에 하이라이트 해두었듯이 **image recognition(이미지 분류/인식)** 입니다. 이는 아주 큰 데이터셋으로 이미 훈련이 된 분류 모델을 사용합니다.

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/imagenet.jpg" width="1000">

[`imageNet`](../c/imageNet.h)은 input(입력)으로 이미지를 받고 output(출력)으로 각 class에 대한 확률값을 얻습니다. **[1000 objects](../data/networks/ilsvrc12_synset_words.txt)** 의 ImageNet ILSVRC dataset으로 최근까지도 훈련돼 오던 GoogleNet 그리고 ResNet-18 모델이 이전에 빌드 하는 과정에서 자동으로 다운받아졌을 겁니다. 다음 [below](#downloading-other-classification-models)을 확인해서 다른 분류 모델도 살펴보시기 바랍니다.

[`imageNet`](../c/imageNet.h) class를 이용하는 예제로써, C++, Python으로 작성된 코드를 제공합니다.: 
- [`imagenet.cpp`](../examples/imagenet/imagenet.cpp) (C++) 
- [`imagenet.py`](../python/examples/imagenet.py) (Python) 

위 샘플들은 이미지, 비디오, 카메라 영상을 분류할 수 있습니다. 지원 되는 input/output stream에 대해 더 많은 정보는 다음 페이지를 참고하세요. [Camera Streaming and Multimedia](aux-streaming.md)

### Jetson 에서의 ImageNet 프로그램 사용

먼저, 몇몇 샘플 이미지에 대해서 `imagenet` 프로그램을 사용해봅시다. 이는 이미지나 이미지들을 load(로드)하고, TensorRT와 `imageNet` 을 이용해서 inference(추론)을 수행합니다. 그리고 분류 결과를 output(출력) 이미지에 overlay(오버래핑) 하여 저장합니다. 이미지들은 해당 프로젝트와 함께 `images/` 디렉토리 아래에 같이 제공됩니다.

프로젝트를 [building](building-repo-2.md)(빌드)한 이후에, terminal의 현재 디렉토리가 `aarch64/bin` 인지 확인하세요.

``` bash
$ cd jetson-inference/build/aarch64/bin
```

다음으로 `imagenet` 프로그램을 이용하여 이미지를 분류해봅시다! 여기서 C++](../examples/imagenet/imagenet.cpp) 혹은 [Python](../python/examples/imagenet.py)을 사용합니다. 만약 도커 컨테이너를 사용하고 있다면 output(결과)를 다음 디렉토리 아래에 저장하는 것을 권장합니다. `images/test` 그러면 호스트 기기에서도 이 이미지들을 다음 디렉토리 `jetson-inference/data/images/test` 아래에서 볼 수 있습니다. 자세한 사항은 다음 페이지를 확인하세요. [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)).


``` bash
# C++
$ ./imagenet images/orange_0.jpg images/test/output_0.jpg     # (기본 모델은 googlenet 입니다.)

# Python
$ ./imagenet.py images/orange_0.jpg images/test/output_0.jpg  # (기본 모델은 googlenet 입니다.)
```

> **note**: 각 모델을 처음 실행할 때는 모델들을 optimize(최적화)하기 때문에 몇 분 정도가 소요됩니다. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 최적화가 된 모델은 disk에 캐싱되며, 이후에 수행될 때는 빠르게 수행됩니다. 

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/imagenet-orange.jpg" width="650">

``` bash
# C++
$ ./imagenet images/strawberry_0.jpg images/test/output_1.jpg

# Python
$ ./imagenet.py images/strawberry_0.jpg images/test/output_1.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/imagenet-strawberry.jpg" width="650">

하나의 이미지 말고도 추가적으로 디렉토리나 이미지의 sequence, 혹인 비디오 파일을 input(입력)으로 줄 수 있습니다. 더 많은 정보는 다음 페이지를 확인하세요. [Camera Streaming and Multimedia](aux-streaming.md). 혹은 `--help` 플래그를 사용하세요.

### 다른 분류 모델 다운로드 받기

빌드 과정에서 (default)기본값으로 GoogleNet과 ResNet-18을 다운로드 받습니다.
그러나 다른 pre-trained models(미리 훈련된 모델)도 사용가능 합니다. [download](building-repo-2.md#downloading-models)

| Network       | CLI argument   | NetworkType enum |
| --------------|----------------|------------------|
| AlexNet       | `alexnet`      | `ALEXNET`        |
| GoogleNet     | `googlenet`    | `GOOGLENET`      |
| GoogleNet-12  | `googlenet-12` | `GOOGLENET_12`   |
| ResNet-18     | `resnet-18`    | `RESNET_18`      |
| ResNet-50     | `resnet-50`    | `RESNET_50`      |
| ResNet-101    | `resnet-101`   | `RESNET_101`     |
| ResNet-152    | `resnet-152`   | `RESNET_152`     |
| VGG-16        | `vgg-16`       | `VGG-16`         |
| VGG-19        | `vgg-19`       | `VGG-19`         |
| Inception-v4  | `inception-v4` | `INCEPTION_V4`   |

> **note**:  추가적으로 모델을 다운로드 받기 위해서는 다음 스크립트를 수행하세요 [Model Downloader](building-repo-2.md#downloading-models)<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>

보통 더 복잡한 모델일 수록 더 좋은 정확도를 보여주지만 속도가 느려집니다.

### 다른 분류 모델 사용하기

`--network` 플래그를 설정함으로써 위에서 제시한 모델중 사용하고자 하는 모델을 (specify)선택할 수 있습니다. 플래그를 설정할 때는 위 표에서 CLI argument 아래에 있는 값들을 주면됩니다. 만약 `--network` 에 아무것도 specify(인자로 넘겨) 주지 않으면 default(기본값)으로 GoogleNet이 선택됩니다.

아래는 ResNet-18을 이용한 예제입니다.:

``` bash
# C++
$ ./imagenet --network=resnet-18 images/jellyfish.jpg images/test/output_jellyfish.jpg

# Python
$ ./imagenet.py --network=resnet-18 images/jellyfish.jpg images/test/output_jellyfish.jpg
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/imagenet_jellyfish.jpg" width="650">

``` bash
# C++
$ ./imagenet --network=resnet-18 images/stingray.jpg images/test/output_stingray.jpg

# Python
$ ./imagenet.py --network=resnet-18 images/stingray.jpg images/test/output_stingray.jpg
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/imagenet_stingray.jpg" width="650">

``` bash
# C++
$ ./imagenet --network=resnet-18 images/coral.jpg images/test/output_coral.jpg

# Python
$ ./imagenet.py --network=resnet-18 images/coral.jpg images/test/output_coral.jpg
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/imagenet_coral.jpg" width="650">

마음껏 다른 모델을 시도해보세요. [Model Downloader](building-repo-2.md#downloading-models) 툴을 이용하여 다른 모델을 다운로드 받을 수 있습니다. 다양한 이미지들이 `images/` 디렉토리 아래 있습니다.

### 비디오 (processing)처리하기

[Camera Streaming and Multimedia](aux-streaming.md) 페이지에서 `imagenet` 프로그램이 다룰 수 있는 다양한 stream(스트림) 타입을 확인할 수 있습니다.
아래는 디스크에 있는 비디오에 대하여 분류하는 예제입니다.:

``` bash
# Download test video (thanks to jell.yfish.us)
$ wget https://nvidia.box.com/shared/static/tlswont1jnyu3ix2tbf7utaekpzcx4rc.mkv -O jellyfish.mkv

# C++
$ ./imagenet --network=resnet-18 jellyfish.mkv images/test/jellyfish_resnet18.mkv

# Python
$ ./imagenet.py --network=resnet-18 jellyfish.mkv images/test/jellyfish_resnet18.mkv
```

<a href="https://www.youtube.com/watch?v=GhTleNPXqyU" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/imagenet-jellyfish-video.jpg width="750"></a>

이제 처음부터 Python과 C++를 이용하여 이미지 분류 프로그램을 코딩해보겠습니다.

##
<p align="right">Next | <b><a href="imagenet-example-python-2.md">Coding Your Own Image Recognition Program (Python)</a></b>
<br/>
Back | <b><a href="building-repo-2.md">Building the Repo from Source</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
