<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="../README.md#hello-ai-world">Back</a> | <a href="aux-docker.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>System Setup</sup></p> 

# Jetson 에 JetPack 설정하기

> **note**:  이미 Jetson Nano나 Xavier, NX에 SD카드로 셋업이 되어있거나, 이미 JetPack이 설치되어있다면 해당 과정을 건너 뛰고 다음 과정인 [`Running the Docker Container`](aux-docker.md)  나 [`Building the Project`](building-repo-2.md) 로 건너뛰셔도 좋습니다.

NVIDIA **[JetPack](https://developer.nvidia.com/embedded/jetpack)** 은 Jetson에서 컴퓨터 비전 어플리케이션을 개발하거나 사용하기 위한 종합 SDK입니다. JetPack은 아래 요소들을 포함하는 OS나  driver를 설치하는 일을 간단히 합니다.

요소:
- L4T Kernel / BSP
- CUDA Toolkit
- cuDNN
- TensorRT
- OpenCV
- VisionWorks
- Multimedia API's

도커 컨테이너를 사용하고 repo를 빌드하기 전에 최신 버전의 JetPack을 설치했는지 꼭 확인하세요.

### Jetson Nano 와 Jetson Xavier NX
Jetson Nano, Xavier, NX의 개발키트를 설치하는 방법으로 가장 권장되는 방법은 **[SD card images](https://developer.nvidia.com/embedded/downloads)** 를 이용해 설치하는 것입니다. 

JetPack 구성 요소가 이미 설치되어 있으며 Windows, Mac 또는 Linux PC에서 플래시할 수 있습니다. 아직 아무것도 설치 돼있지 않다면 각 Jetson에 따라 제공되는 가이드를 따라 수행하세요.

* [Jetson Nano Developer Kit 으로 시작하기](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
* [Jetson Nano 2GB Developer Kit 으로 시작하기](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit)
* [Jetson Xavier NX 사용자 가이드](https://developer.nvidia.com/embedded/downloads#?search=Jetson%20Xavier%20NX%20Developer%20Kit%20User%20Guide) 

### Jetson TX1/TX2 and AGX Xavier

다른 jetson들은 다음 [NVIDIA SDK Manager](https://developer.nvidia.com/embedded/dlc/nv-sdk-manager)를 Ubuntu 16.04 x86_64 나 Ubuntu 18.04 x86_64가 설치돼있는 host PC에서 다운받아 플래시 해야합니다. Micro-USB나 USB-C 를 host PC에 연결하고 jetson을 Recovery Mode에 진입하게 하세요. 

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/nvsdkm.png" width="800">

상세한 내용은 다음 문서를 확인하세요. **[NVIDIA SDK Manager Documentation](https://docs.nvidia.com/sdk-manager/index.html)**.

### 프로젝트 시작

프로젝트 앞서서 사용할 수 있는 방법이 2가지가 있습니다.

* 미리 빌드된 [Docker Container](aux-docker.md)를 수행하는 방법
* 소스코드로부터 직접 빌드하기 [Build the Project from Source](building-repo-2.md)

도커 컨테이너를 사용하는 것이 처음에는 가능한 빨리 수행하기에 권장되는 방법입니다. (Pytorch도 이미 설치돼있습니다.) 그러나 Jetson에 익숙하다면 직접 컴파일하여 빌드하는 것도 그렇게 복잡하지 않습니다.

##
<p align="right">Next | <b><a href="building-repo-2.md">Building the Project from Source</a></b>
<br/>
Back | <b><a href="../README.md#hello-ai-world">Overview</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
