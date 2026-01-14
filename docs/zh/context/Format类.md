# Format类

数据格式的枚举类，提供了GE的data format定义，方便实现converter函数时调用。数据格式（format）是用于描述一个多维Tensor的轴的业务语义，表示数据的物理排布格式，定义了解读数据的维度，例如1D、2D、3D、4D、5D等。

Format类的具体定义如下：

```python
class Format(Enum):
    FORMAT_UNDEFINED = -1
    FORMAT_NCHW = 0
    FORMAT_NHWC = 1
    FORMAT_ND = 2
    FORMAT_NC1HWC0 = 3
    FORMAT_FRACTAL_Z = 4
    FORMAT_NC1HWC0_C04 = 12
    FORMAT_HWCN = 16
    FORMAT_NDHWC = 27
    FORMAT_FRACTAL_NZ = 29
    FORMAT_NCDHW = 30
    FORMAT_NDC1HWC0 = 32
    FORMAT_FRACTAL_Z_3D = 33
    FORMAT_NC = 35
    FORMAT_NCL = 47
```

数据格式一般形式为“FORMAT\__XXXX_”，数据格式中维度含义：N（Batch）表示批量大小、H（Height）表示特征图高度、W（Width）表示特征图宽度、C（Channels）表示特征图通道、D（Depth）表示特征图深度。

