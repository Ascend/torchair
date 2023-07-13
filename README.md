torchair实现将torch的FX图转换为GE计算图，并提供了GE计算图的编译与执行接口。

# 编译准备

配置编译环境，执行配置命令
> 配置只需要进行一次，用于获取pytorch的编译选项（如当前的torch是否开启ABI）及Ascend sdk的安装路径（如果需要在本地CPU上进行调试）。
```shell
bash ./configure
```
默认情况下，执行上述命会弹出如下的交互式会话窗口
> 您的会话可能有所不同。

```BASH
Please specify the location of python with available torch 2.x installed. [Default is /usr/bin/python3]
(You can make this quiet by set env [TARGET_PYTHON_PATH]):
```

此时，要求您输入安装了 Torch 2.x 版本的python解释器路径，如果默认路径是正确的，直接回车，否则请输入正确的 python 解释器路径。
> 您可以通过设置 `TARGET_PYTHON_PATH` 环境变量，来抑制交互式窗口弹出，但是要确保路径是有效的，否则，仍然会要求您输入正确的 python 解释器路径。

键入后，会耗费几秒钟以确保您的输入是有效的，接着，会弹出下面的交互式窗口

```BASH
Specify the location of ascend sdk for debug on localhost or leave empty.
(You can make this quiet by set env [ASCEND_SDK_PATH]):
```

如果您不需要在本地CPU调试，可以直接回车跳过。否则，需要输入昇腾处理器开发套件的安装路径（需指定至opensdk/opensdk目录）。

> 您可以通过设置 `ASCEND_SDK_PATH` 环境变量指定 SDK 目录 或 设置 `NO_ASCEND_SDK` 环境变量指定不需要 SDK 来抑制交互式窗口弹出。

键入后，等待配置完成。

# 编译与安装

```shell
mkdir build
cd build
make ..
make torchair -j8
```

编译完成后，会在`build/dist/dist/`目录下生成名为torchair-{version}-py3-none-any.whl的安装包文件。

您可以直接使用pip安装该安装包，或者使用make命令安装至您configure时指定的python环境中。
```shell
make install_torchair
```

# 执行测试

> 如果您在配置时未指定Ascend sdk的安装路径，则无法执行CPU上的调试，需要在NPU环境上进行测试。

CPU调试时，需要设置LD_LIBRARY_PATH到生成的fake so文件目录以及sdk目录

> tools/env.sh会根据配置生成对应的LD_LIBRARY_PATH（如果您在配置时指定了Ascend sdk安装路径）
```shell
source tools/env.sh

python3 examples/example.py
```

# 关于CI编译

工程目录下的build.sh用于对接CI编译，但是您也可以通过其在本地执行编译和UT/ST测试。

## 查看帮助
```shell
./build.sh -h
```

## 编译安装包
```shell
./build.sh -c
```
编译完成后，会在`output/`目录下生成名为torchair-{version}-py3-none-any.whl的安装包文件。
## 执行UT测试
> 本地执行UT测试需要设置环境变量`ASCEND_CUSTOM_PATH`,将其指定至Ascend sdk的安装路径（指定至ai_cann_x86目录）
```shell
./build.sh -u
```

## 执行ST测试
> 本地执行ST测试需要设置环境变量`ASCEND_CUSTOM_PATH`,将其指定至Ascend sdk的安装路径（指定至ai_cann_x86目录）
```shell
./build.sh -s
```

## 查看覆盖率报告
UT或ST执行通过后，会在coverage目录下生成覆盖率文件coverage.info，如果您想要查看覆盖率报告，可以执行如下命令
```shell
genhtml coverage.info -o coverage_report
```
