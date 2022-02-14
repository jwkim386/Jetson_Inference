<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="jetpack-setup-2.md">Back</a> | <a href="building-repo-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>System Setup</sup></p>  

# Docker Container Run(수행하기)

해당 프로젝트를 진행하기 위한 미리 빌드된 도커 컨테이너 이미지는 [DockerHub](https://hub.docker.com/r/dustynv/jetson-inference/tags)에 업로드 돼있습니다. 혹은 직접 빌드를 진행할 수도 있습니다. [Build the Project from Source](building-repo-2.md).

아래는 현재 사용 가능한 컨테이너들의 태그들입니다.

| Container Tag                                                                           | L4T version |          JetPack version         |
|-----------------------------------------------------------------------------------------|:-----------:|:--------------------------------:|
| [`dustynv/jetson-inference:r32.6.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.6.0 | JetPack 4.6 |
| [`dustynv/jetson-inference:r32.5.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.5.0 | JetPack 4.5 |
| [`dustynv/jetson-inference:r32.4.4`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.4.4 | JetPack 4.4.1 |
| [`dustynv/jetson-inference:r32.4.3`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.4.3 | JetPack 4.4 |


> **note:** Jetson에 설치한 Jetpack-L4T의 버전과 위에서 제시한 L4T version이 일치해야합니다. 만약 다른 버전의 JetPack-L4T를 설치했다면 JetPack을 최신 버전으로 업그레이드 하거나 직접 빌드하여 사용하는 방법이 있습니다 [Build the Project from Source](docs/building-repo-2.md) to compile the project directly.  

위 도커 컨테이너들은 [`l4t-pytorch`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch) 베이스 컨테이너로 사용하고 있습니다. 따라서 tranfer learning과 re-training에 대한 지원을 포함하고 있습니다. 

## 컨테이너 launch 하기

여러 마운트들과 장치들이 컨테이너를 run 하기 원하기 때문에 [`docker/run.sh`](../docker/run.sh) 스크립트를 사용하여 컨테이너를 run(수행)하는 것을 권장합니다.

```bash
$ git clone --recursive https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ docker/run.sh
```

> **note:**  도커 스크립트와 컨테이너 안에 마운트된 데이터 디렉토리 구조 때문에 호스트 디바이스에 프로젝트를 clone 해야합니다. 

[`docker/run.sh`](../docker/run.sh) 는 자동으로 알맞은 컨테이터 tag를 현재 설치된 JetPack-L4T 버전을 기반으로 도커 허브로 부터 pull(다운로드) 해옵니다. 그리고 데이터 디렉토리와 사용할 기기들 사용할 수 있도록 마운트합니다. 따라서 컨테이너 안에서 cameras/display/ect 로부터 카메라를 사용할 수 있습니다. 그리고 [download DNN models](building-repo-2.md#downloading-models) 하도록 입력창이 나타납니다. 만약 아직 아무런 입력창이 나타나지 않는다면 아직은 도커 컨테이너가 설치 중인 것입니다. 금방 끝납니다. 

### Mounted Data Volumes

참고로, 아래 경로들이 jetson 장치 자체에서 컨테이너로 마운트됩니다.

* `jetson-inference/data` (network models이 저장됨, serialized TensorRT engines, and test images)
* `jetson-inference/python/training/classification/data` (classification training datasets 이 저장됨)
* `jetson-inference/python/training/classification/models` (Pytorch로 훈련된 classification models 이 저장됨)
* `jetson-inference/python/training/detection/ssd/data` (객체 검출 training datasets 이 저장됨)
* `jetson-inference/python/training/detection/ssd/models` (pytorch로 훈련된 객체 검출 models 이 저장됨)

위 마운트된 (volumes)디렉토리들은 컨테이너 밖에 저장돼있기 때문에 컨테이너가 꺼져도 (shut down) 삭제되지 않습니다. 

만약 직접 다른 디렉토리를 컨테이너 안에 마운트하고 싶다면 다음과 같은 인자를 `--volume HOST_DIR:MOUNT_DIR` [`docker/run.sh`](../docker/run.sh) 명령에 전달할 수 있습니다.

```bash
$ docker/run.sh --volume /my/host/path:/my/container/path    # these should be absolute paths
```

더 많은 정보가 필요하면 다음 파일을 직접 확인할 수 있습니다. [`docker/run.sh`](../docker/run.sh)

## 어플리케이션 (Run)수행

컨테이너가 올라가 있고 (running)실행되고 있다면, 이제는 튜토리얼의 예제 프로그램들을 컨테이너 내부에서 아래와 같이 실행시켜볼 수 있습니다. 

```bash
# cd build/aarch64/bin
# ./video-viewer /dev/video0
# ./imagenet images/jellyfish.jpg images/test/jellyfish.jpg
# ./detectnet images/peds_0.jpg images/test/peds_0.jpg
# (press Ctrl+D to exit the container)
```

> **note:** 만약 수행 결과 이미지를 저장하고 싶다면, `images/test` 해당 디렉토리 아래에 저장하는 것을 권장합니다. 그러면 해당 이미지를 호스트 기기의 다음 디렉토리에서 `jetson-inference/data/images/test` 쉽게 볼 수 있기 때문입니다. 

## 컨테이너 빌드하기

만약 Hello AI World 튜토리얼을 따라하는 중이라면 해당 섹션을 넘어가도 괜찮습니다. 그러나 직접 새로운 컨테이너 빌드하거나 re-build(다시 빌드) 하고 싶다면 다음 스크립트를 수행할 수 있습니다. [`docker/build.sh`](../docker/build.sh). 해당 스크립트는 이 프로젝트의 [`Dockerfile`](../Dockerfile) 파일을 빌드합니다.

```bash
$ docker/build.sh
```

>  **note:** 먼저 디폴트 `docker-runtime` 을 nvidia에 설정해두어야합니다. 자세한 사항은 다음을 참고하세요. [here](https://github.com/dusty-nv/jetson-containers#docker-default-runtime)

Dockerfile을 다음과 `FROM dustynv/jetson-inference:r32.4.3` 같이 수정하여 직접 생성하고 싶은 컨테이너의 (base)베이스를 이 컨테이너 위에 올릴 수 있습니다. 

## Getting Started

도커 컨테이너 안에서 프로젝트를 진행하기로 했다면, 다음을 진행하세요. [ImageNet으로 이미지 분류하기](imagenet-console-2.md).<br>
그렇지 않고 소스코드로 직접 수행하기로 했다면, 다음을 참조하세요. [소스코드로부터 프로젝트 빌드하기](building-repo-2.md).<br>
 
##
<p align="right">Next | <b><a href="building-repo-2.md">Building the Project from Source</a></b>
<br/>
Back | <b><a href="jetpack-setup-2.md">Setting up Jetson with JetPack</a></p>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

