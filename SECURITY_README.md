# 安全声明
## 系统安全加固
1. 建议用户在系统中配置开启ASLR（级别2），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：
    ```
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

## 运行用户建议
出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用TorchAir。

## 文件权限控制
1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控。涉及场景如TorchAir安装目录权限管控、多用户使用共享数据集权限管控、GE图dump权限管控等场景，管控权限可参考表1进行设置。
3. TorchAir中集成GE图dump工具，使用时会在本地生成GE图，文件权限默认600，用户可根据实际需求对生成文件权限进行进阶管控。

**表1 文件（夹）各场景权限管控推荐最大值**
| 类型          | linux权限参考最大值 |
| --------------- | --------------------|
| 用户主目录                          |    750（rwxr-x---）                |
| 程序文件（含脚本文件、库文件等）      |    550（r-xr-x---）                |
| 程序文件目录                        |    550（r-xr-x---）                |
| 配置文件                            |    640（rw-r-----）                |
| 配置文件目录                        |    750（rwxr-x---）                |
| 日志文件（记录完毕或者已经归档）      |    440（r--r-----）                |
| 日志文件（正在记录）                 |    640（rw-r-----）                |
| 日志文件记录                        |    750（rwxr-x---）                |
| Debug文件                          |    640（rw-r-----）                |
| Debug文件目录                      |    750 (rwxr-x---)                 |
| 临时文件目录                       |     750（rwxr-x---）                |
| 维护升级文件目录                    |    770（rwxrwx---）                |
| 业务数据文件                       |     640（rw-r-----）                |
| 业务数据文件目录                   |     750（rwxr-x---）                |
| 密钥组件、私钥、证书、密文文件目录   |     700（rwx------）                |
| 密钥组件、私钥、证书、加密密文      |     600（rw-------）                |
| 加解密接口、加解密脚本             |     500（r-x------）                |


## 调试工具声明

1. TorchAir内集成生成GE图dump数据调试工具：
    - 集成原因：对标pytorch原生支持能力，提供NPU编译后端图dump能力，加速图模式开发调试过程。
    - 使用场景：默认不开启，如用户使用pytorch图模式且需要进行图分析时，可在模型训练脚本中调用生成GE图dump数据接口生成dump数据。
    - 风险提示：使用该功能会在本地生成GE图，用户需加强对相关dump数据的保护，请在图模式调试阶段使用该能力，调试完毕后及时关闭。

2. TorchAir内集成生成数据dump调试工具：
    - 集成原因：提供NPU图模式中间算子数据dump功能，用于比对精度，加速图模式开发调试过程。
    - 使用场景：默认不开启，如用户使用pytorch图模式且需要进行精度分析时，可在模型启动脚本中调用数据dump接口生成dump数据。
    - 风险提示：使用该功能会在本地生成计算数据，用户需加强对相关dump数据的保护，请在图模式调试阶段使用该能力，调试完毕后及时关闭。


## 构建安全声明

1. TorchAir在源码编译安装过程中，会下载依赖三方库并执行构建shell脚本，编译过程中会产生临时编译目录和程序文件。用户可根据需要，对源代码目录中的文件及文件夹进行权限管控，降低安全风险。


## 运行安全声明

1. 建议用户结合运行资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
2. PyTorch、torch_npu和TorchAir在运行异常时会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括通过设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。


## 公网地址声明

代码涉及公网地址参考[public_address_statement.md](https://gitee.com/ascend/torchair/blob/master/public_address_statement.md)


## 公开接口声明

AscendPyTorch是PyTorch适配插件，TorchAir是为AscendPyTorch提供图模式能力的扩展库，支持用户使用PyTorch在昇腾设备上进行图模式的训练和推理。

参考[PyTorch社区公开接口规范](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation)，  TorchairAir提供了对外的自定义接口。TorchAir提供在昇腾设备上的编译后端以对接Pytorch的原生torch.compile接口，因此，TorchAir对外提供了接口来实现此功能，具体接口可参考[《README》](https://gitee.com/ascend/torchair/blob/master/README.md)的《torchair常用类和公开接口介绍》章节。如果一个函数看起来符合公开接口的标准且在文档中有展示，则该接口是公开接口。否则，使用该功能前可以在社区询问该功能是否确实是公开的或意外暴露的接口，因为这些未暴露接口将来可能会被修改或者删除。

TorchAir项目采用C++和Python联合开发，当前正式接口只提供Python接口，在TorchAir的二进制包中动态库不直接提供服务，暴露的接口为内部使用，不建议用户使用。

## 通信安全加固

TorchAir在运行时依赖PyTorch及torch_npu，用户需关注通信安全加固，具体方式可参考torch_npu[通信安全加固](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)。


## 通信矩阵

TorchAir在运行时依赖PyTorch及torch_npu，涉及通信矩阵，具体方式可参考torch_npu[通信矩阵](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。