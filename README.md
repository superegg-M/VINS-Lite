# Vins Lite
**作者**：许家仁

**描述**：
该代码基于 VINS-Mono与VINS-Course，不依赖 ROS。在代码中实现了一个基于Eigen的轻量级图优化求解器，用于求解VINS的后端优化问题，该求解器目前不支持对顶点和边的删除操作，每次优化前必选重新构建问题。该代码支持 Ubuntu.

___

### 安装依赖项：

1. pangolin: <https://github.com/stevenlovegrove/Pangolin>

2. opencv-3.4.0

3. Eigen-3.3.9

4. Ceres-1.14.0: 用于VINS的初始化中的SfM。

5. boost

___

### 编译代码

```c++
mkdir vins_lite
cd vins_lite
git clone git@github.com:CainHu/VINS-Lite.git
mkdir build 
cd build
cmake ..
make -j4
```

___

### 运行
#### 1. CurveFitting Example to Verify Our Solver.
```c++
cd build
../bin/testCurveFitting 
```

#### 2. VINs-Mono on Euroc Dataset
```c++
cd build
../bin/run_euroc /home/cain/mav0/ ../config/
```
![vins](doc/vins.gif)

#### 3. VINs-Mono on Simulation Dataset

可以使用以下代码生成的vio数据来进行调试。

<https://github.com/HeYijia/vio_data_simulation>

___

### Licence

The source code is released under GPLv3 license.

如果发现了代码中存在的bug以及可改进的方向, 请联系: Cain Hu <cainhsui@gmail.com>.

___

### 参考

[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) 
 [VINS-Course](https://github.com/HeYijia/VINS-Course)

