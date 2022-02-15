<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="posenet.md">Back</a> | <a href="pytorch-transfer-learning.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Mono Depth</sup></s></p>

# DepthNet으로 하는 Monocular(하나의 카메라로 하는) Depth 

물체가 얼마나 멀리 떨어져있는지를 나타내는 Depth sensing은 맵핑이나, 내비게이션, 장애물 검출등에 있어서 유용합니다. 그러나 이전에는 스테레오 카메라나 RGB-D 카메라가 필요했습니다. 그러나 이제는 딥러닝이 등장하면서 하나의 카메라(monocular)로 상대적인 거리를 추정할 수 있게 되었습니다. 이를 mono depth 라고합니다. 자세한 내용은 다음 페이지를 참고하세요. [MIT FastDepth](https://arxiv.org/abs/1903.03273)

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/depthnet-0.jpg">

The [`depthNet`](../c/depthNet.h) 는 한 장의 컬러 이미지를 input(입력)으로 받고 depth map(깊이 맵)을 output(출력)으로 합니다. 해당 depth map(깊이 맵)은 사람이 보기 좋게 색처리가 돼있습니다. 하지만 색처리되지 않은 depth map(깊이 맵)의 depth(깊이 값)에도 접근할 수 있습니다. 이는 [Python](https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.inference.html#depthNet) and [C++](../c/depthNet.h). 으로 사용 가능합니다.

실행 예제로 C++, Python 코드가 제공됩니다.

- [`depthnet.cpp`](../examples/depthnet/depthnet.cpp) (C++) 
- [`depthnet.py`](../python/examples/depthnet.py) (Python) 

위 샘플들은 이미지, 비디오, 카메라 스트리밍을 infer(추론)할 수 있습니다. 더 자세한 input/output 스트림에 대한 정보는 다음 페이지를 참고하세요. [Camera Streaming and Multimedia](aux-streaming.md)

## 이미지에 대한 Mono Depth 

먼저 이미지에 대해 `depthnet`을 적용해봅시다. input/output 경로 외에도 커맨드 라인에서의 옵션 플래그가 존재합니다.:

- optional `--network` 플래그는 사용하고자 하는 모델을 바꿀 수 있습니다. (권장되는 default(기본값)은 `fcn-mobilenet`).
- optional `--visualize` 플래그는 `input`, `depth`의 comma-separated(컴마로 구분되는) 조합입니다.
	- The default(기본값)은  `--visualize=input,depth` 이며 이는 input 이미지와 depth를 나란히 출력합니다.
	- depth 이미지만 보고싶다면, `--visualize=depth`를 사용하세요.
- optional `--depth-size` 플래그는 depth map의 input 이미지에 대한 상대적 스케일 값을 변경할 수 있습니다. (the default(기본값)은 `1.0`)
- optional `--filter-mode` upsampling을 할 때, 사용할 필터링 방법을 결정합니다. `point` 나 `linear` 중에 선택할 수 있습니다. (the default(기본값)은 `linear`)
- optional `--colormap` 플래그는 컬러맵을 정합니다. (the default(기본값)은 `viridis_inverted`)

만약 [Docker container](aux-docker.md)를 사용하고 있다면 결과 이미지를 `images/test` 디렉토리 아래에 저장하는 것을 권장합니다. 그래야 해당 이미지들이 호스트 기기에서 `jetson-inference/data/images/test` 디렉토리 아래를 확인하여 위 결과 이미지들을 볼 수 있기 때문입니다. (자세한 정보는 다음 페이지를 참고하세요. [Mounted Data Volumes](aux-docker.md#mounted-data-volumes))

아래 예제는 실내에서 수행한 mono depth 입니다.

``` bash
# C++
$ ./depthnet "images/room_*.jpg" images/test/depth_room_%i.jpg

# Python
$ ./depthnet.py "images/room_*.jpg" images/test/depth_room_%i.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/depthnet-room-0.jpg">

> **note**: 모델을 처음 수행시키면 TensorRT가 모델을 optimize(최적화)하기 때문에 몇 분 정도 소요될 수 있습니다. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 위 최적화된 모델은 disk에 캐싱되기 때문에 이후에 모델을 불러올 때는 빠르게 불러와집니다.

아래는 바깥에서 mono depth를 수행한 예제 입니다.

``` bash
# C++
$ ./depthnet "images/trail_*.jpg" images/test/depth_trail_%i.jpg

# Python
$ ./depthnet.py "images/trail_*.jpg" images/test/depth_trail_%i.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/depthnet-trail-0.jpg">

## 비디오에서 Mono Depth 

비디오에서 mono depth estimation을 수행하기 위해서 라이브 카메라나 비디오의 디바이스 혹은 파일 패스를 전달해야합니다. 자세한 정보는 다음 페이지를 확인하세요. [Camera Streaming and Multimedia](aux-streaming.md)

``` bash
# C++
$ ./depthnet /dev/video0     # csi://0 if using MIPI CSI camera

# Python
$ ./depthnet.py /dev/video0  # csi://0 if using MIPI CSI camera
```

<a href="https://www.youtube.com/watch?v=3_bU6Eqb4hE" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/depthnet-video-0.jpg width="750"></a>

> **note**:  만약 화면이 output을 표현하기에 너무 작으면 `--depth-scale=0.5` 옵션을 주어 사이즈를 줄일 수 있습니다.  <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of the depth image, or reduce the size of the camera with `--input-width=X --input-height=Y`


## Depth Field 얻기

만약 raw(처리되지 않은) depth map에 접근하고 싶으면 `depthNet.GetDepthField()` 로 이를 받아올 수 있다. 이는 보톤 224x224 보다 작은 하나의 채널을 갖는 floating point 이미지를 return합니다. 이는 모델의 순수한 결과입니다. 반면에 visualization에 사용된 색처리가된 depth 이미지는 원본 이미지의 크기에 맞춰 upsampling된 이미지 입니다. 혹은   `--depth-size`에 의해 적당한 크기로 스케일링된 크기일 것입니다.

아래는 Python, C++ 로 작성된 pseudo 코드 입니다.:

#### Python

``` python
import jetson.inference
import jetson.utils

import numpy as np

# load mono depth network
net = jetson.inference.depthNet()

# depthNet re-uses the same memory for the depth field,
# so you only need to do this once (not every frame)
depth_field = net.GetDepthField()

# cudaToNumpy() will map the depth field cudaImage to numpy
# this mapping is persistent, so you only need to do it once
depth_numpy = jetson.utils.cudaToNumpy(depth_field)

print(f"depth field resolution is {depth_field.width}x{depth_field.height}, format={depth_field.format})

while True:
    img = input.Capture()	# assumes you have created an input videoSource stream
    net.Process(img)
    jetson.utils.cudaDeviceSynchronize() # wait for GPU to finish processing, so we can use the results on CPU
	
    # find the min/max values with numpy
    min_depth = np.amin(depth_numpy)
    max_depth = np.amax(depth_numpy)
```

#### C++

``` cpp
#include <jetson-inference/depthNet.h>

// load mono depth network
depthNet* net = depthNet::Create();

// depthNet re-uses the same memory for the depth field,
// so you only need to do this once (not every frame)
float* depth_field = net->GetDepthField();
const int depth_width = net->GetDepthWidth();
const int depth_height = net->GetDepthHeight();

while(true)
{
    uchar3* img = NUL;
    input->Capture(&img);  // assumes you have created an input videoSource stream
    net->Process(img, input->GetWidth(), input->GetHeight());

    // wait for the GPU to finish processing
    CUDA(cudaDeviceSynchronize()); 
	
    // you can now safely access depth_field from the CPU (or GPU)
    for( int y=0; y < depth_height; y++ )
        for( int x=0; x < depth_width; x++ )
	       printf("depth x=%i y=%i -> %f\n", x, y, depth_map[y * depth_width + x]);
}
```

mono depth 이미지만을 가지고 absolute(절대) 거리를 측정하면 부정확한 결과가 나옵니다. 보통 relative(상대적인)거리가 더 정확합니다. raw dapth 이미지는 이미지에 따라 그 범위가 천차만별입니다. 그래서 보통 자동으로 이들의 범위를 다시 계산하여 조정합니다. visualization 히스토그램을 equlization(평탄화)하는 과정에서 depth의 값의 분포를 평평하게 만듭니다. 

다음으로는 [Transfer Learning](pytorch-transfer-learning.md) 에 대한 개념을 배우도록 하겠습니다. 그리고 커스텀 딥러닝 모델을 Pytorch, Jetson을 이용해 훈련시켜보겠습니다.

##
<p align="right">Next | <b><a href="pytorch-transfer-learning.md">Transfer Learning with PyTorch</a></b>
<br/>
Back | <b><a href="posenet.md">Pose Estimation with PoseNet</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
