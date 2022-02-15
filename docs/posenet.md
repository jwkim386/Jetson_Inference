* Auth: 김준호
* Data: 2022-02-15

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="segnet-camera-2.md">Back</a> | <a href="depthnet.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Pose Estimation</sup></s></p>

# PoseNet 으로 하는 Pose Estimation
Pose estimaion은 link라 불리는 skeleton topology(해부학적 위상)를 구성하는 다양한 신체의 일부분들의 (keypoint라고 합니다.) 위치를 찾는 것을 말합니다. Pose estimaion은 제스쳐, AR/VR, HMI (human/machine interface) 그리고 자세 교정등과 같은 다양한 분야에 적용됩니다. 하나의 프레임에 여러 사람이 있어도 한 번에 몸과 손을 검출할 수 있는 [Pre-trained models](#pre-trained-pose-estimation-models) 이 준비돼 있습니다.  

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-0.jpg">

The [`poseNet`](../c/poseNet.h) 은 이미지를 input(입력)으로 받고 각 포즈들의 리스트를 output(출력)으로 합니다. 각 객체들의 pose는 검출된 keypoints와 이들의 위치, 그리고 keypoints 사이의 link들이 포함됩니다. 이들 중에서 필요한 것만 (query)골라 사용할 수 있습니다. [`poseNet`](../c/poseNet.h) 은  [Python](https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.inference.html#poseNet) and [C++](../c/poseNet.h) 코드로 사용 가능합니다.

아래는 C++, Python으로 제공된 `poseNet` 예제 프로그램입니다.

- [`posenet.cpp`](../examples/posenet/posenet.cpp) (C++) 
- [`posenet.py`](../python/examples/posenet.py) (Python) 

위 예제들은 이미지, 비디오, 카메라 스트리밍에서 여러명의 사람들에 대하여 검출해낼 수 있습니다. 더 자세한 input/output 스트림에 대한 정보는 다음 페이지를 참고하세요. [Camera Streaming and Multimedia](aux-streaming.md)

## Images 에서의 Pose Estimation

첫 째로, 몇 이미지에 대해서 `posenet`을 수행시켜봅시다. 커맨드 라인에서 input/output 경로 말고도 인자로 다음과 같은 옵션들을 더 줄 수 있습니다.:

- optional `--network` 플래그는 사용할 [pose model](#pre-trained-pose-estimation-models) 을 바꿀 수 있습니다. 
- optional `--overlay` 플래그는 `box`, `links`, `keypoints`, 혹은 `none` 을 comma-separated(컴마로 구분하여) 옵션을 정할 수 있습니다. 
	- The default(기본값) 은 `--overlay=links,keypoints` 이고 이는 keypoints에는 원을, 링크에는 라인을 그립니다.
- optional `--keypoint-scale` keypoint에 그려지는 circle의 반지름을 결정하는 값입니다. (the default(기본값)은 `0.0052`)
- optional `--link-scale` 값은 라인의 두계를 결정합니다. (the default(기본값)은 `0.0013`)
- optional `--threshold` 검출에 최소값을 결정합니다. (the default(기본값)은 `0.15`).  

만약 [Docker container](aux-docker.md) 를 사용하고 있다면 output(출력) 이미지를 `images/test` 아래에 저장하는 것을 권장합니다. 그래야 호스트 기기에서 해당 이미지들을 `jetson-inference/data/images/test` 디렉토리 아래에서 볼 수 있기 때문입니다. (더 많은 정보를 원하시면 다음 페이지를 참고하세요. [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)).

default(기본값인) Pose-ResNet18-Body 모델을 이용한 human pose estimation의 예제 입니다.:

``` bash
# C++
$ ./posenet "images/humans_*.jpg" images/test/pose_humans_%i.jpg

# Python
$ ./posenet.py "images/humans_*.jpg" images/test/pose_humans_%i.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-1.jpg">

> **note**: 모델을 처음 (run)수행시키면 TensorRT가 이를 optimize(최적화)하는데 시간이 몇 분 정도 소요됩니다. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 이렇게 최적화된 모델은 disk에 캐싱되므로, 이후의 실행에는 금방 실행됩니다.

`"images/peds_*.jpg"` 와 같은 이미지들이 테스트용으로 준비돼있습니다. 

## 비디오로부터 Pose Estimation하기 

Pose estimation을 위해 라이브 카메라나 비디오 파일의 경로를 주어야합니다. 자세한 정보는 다음 페이지를 참고하세요. [Camera Streaming and Multimedia](aux-streaming.md)

``` bash
# C++
$ ./posenet /dev/video0     # csi://0 if using MIPI CSI camera

# Python
$ ./posenet.py /dev/video0  # csi://0 if using MIPI CSI camera
```

<a href="https://www.youtube.com/watch?v=hwFtWYR986Q" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-video-body.jpg width="750"></a>

``` bash
# C++
$ ./posenet --network=resnet18-hand /dev/video0

# Python
$ ./posenet.py --network=resnet18-hand /dev/video0
```

<a href="https://www.youtube.com/watch?v=6NL_IE44vRE" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-video-hands.jpg width="750"></a>

## Pre-trained(훈련된) Pose Estimation 모델

아래 테이블은 [download](building-repo-2.md#downloading-models) 가능한 pre-trained(훈련된) 모델들입니다. 이는 `--network` 인자와 관련돼있으며 CLI argument중 하나의 이름으로 전달할 수 있습니다.:

| Model                   | CLI argument       | NetworkType enum   | Keypoints |
| ------------------------|--------------------|--------------------|-----------|
| Pose-ResNet18-Body      | `resnet18-body`    | `RESNET18_BODY`    | 18        |
| Pose-ResNet18-Hand      | `resnet18-hand`    | `RESNET18_HAND`    | 21        |
| Pose-DenseNet121-Body   | `densenet121-body` | `DENSENET121_BODY` | 18        |

> **note**:  다른 모델을 다운로드 받으려면 다음을 실행시키시면 됩니다. [Model Downloader](building-repo-2.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>

원하는 모델을 불러오기 위해 커맨드 라인에서 `--network`에 위 테이블의 CLI arguments에 원하는 모델의 대응되는 이름을 전달할 수 있습니다. 기본값은 Pose-ResNet18-Body 이고  `--network` 를 사용하지 않으면 해당 값으로 설정됩니다.

## Working with Object Poses

pose keypoints에 직접 접근하여 값을 사용하고 싶다면 `poseNet.Process()` 함수가 `poseNet.ObjectPose` structure 의 리스트를 반환한다는걸 아셔야합니다. 각 ObjectPose는 하나의 객체에 대한 (예를 들어 한 명의 사람) 각 keypointes들을 포함하고 있습니다. 자세한 정보는 다음 페이지를 참고하세요. [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#poseNet) and [C++](../c/poseNet.h) 

아래는 `left_shoulder(왼쪽 어깨)` 와 `left_wrist(왼쪽 손목)` keypoints의 벡터를 이용하여 어떤 사람이 무엇을 가리키고 있는 2D direction(방향)을 구하고자하는 pseudocode입니다.:

``` python
poses = net.Process(img)

for pose in poses:
    # find the keypoint index from the list of detected keypoints
    # you can find these keypoint names in the model's JSON file, 
    # or with net.GetKeypointName() / net.GetNumKeypoints()
    left_wrist_idx = pose.FindKeypoint('left_wrist')
    left_shoulder_idx = pose.FindKeypoint('left_shoulder')

    # if the keypoint index is < 0, it means it wasn't found in the image
    if left_wrist_idx < 0 or left_shoulder_idx < 0:
        continue
	
    left_wrist = pose.Keypoints[left_wrist_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]

    point_x = left_shoulder.x - left_wrist.x
    point_y = left_shoulder.y - left_wrist.y

    print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")
```

간단한 예제를 살펴보았는데요, 더 많은 키포인트들을 살펴보고 이들을 적절히 조합하여 더 나은 무언가를 만들어낼 수 있습니다. 제스쳐를 분류하는 더 나은 딥러닝 기술이 있는데요 가령 [`trt_hand_pose`](https://github.com/NVIDIA-AI-IOT/trt_pose_hand) 와 같은 것들이 있습니다.

	
##
<p align="right">Next | <b><a href="depthnet.md">Monocular Depth Estimation</a></b>
<br/>
Back | <b><a href="segnet-camera-2.md">Running the Live Camera Segmentation Demo</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
