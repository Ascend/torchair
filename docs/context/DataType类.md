# DataType类

数据类型的枚举类，提供了GE的data type定义，方便实现converter函数时调用。DataType类的具体定义如下：

```python
class DataType:
    DT_FLOAT = 0            # float type
    DT_FLOAT16 = 1          # fp16 type
    DT_INT8 = 2             # int8 type
    DT_INT16 = 6            # int16 type
    DT_UINT16 = 7           # uint16 type
    DT_UINT8 = 4            # uint8 type
    DT_INT32 = 3            # int32 type
    DT_INT64 = 9            # int64 type
    DT_UINT32 = 8           # unsigned int32
    DT_UINT64 = 10          # unsigned int64
    DT_BOOL = 12            # bool type
    DT_DOUBLE = 11          # double type
    DT_STRING = 13          # string type
    DT_DUAL_SUB_INT8 = 14   # dual output int8 type
    DT_DUAL_SUB_UINT8 = 15  # dual output uint8 type
    DT_COMPLEX64 = 16       # complex64 type
    DT_COMPLEX128 = 17      # complex128 type
    DT_QINT8 = 18           # qint8 type
    DT_QINT16 = 19          # qint16 type
    DT_QINT32 = 20          # qint32 type
    DT_QUINT8 = 21          # quint8 type
    DT_QUINT16 = 22         # quint16 type
    DT_RESOURCE = 23        # resource type
    DT_STRING_REF = 24      # string ref type
    DT_DUAL = 25            # dual output type
    DT_VARIANT = 26         # dt_variant type
    DT_BF16 = 27            # bf16 type
    DT_UNDEFINED = 28       # Used to indicate a DataType field has not been set.
    DT_INT4 = 29            # int4 type
    DT_UINT1 = 30           # uint1 type
    DT_INT2 = 31            # int2 type
    DT_UINT2 = 32           # uint2 type
    DT_COMPLEX32 = 33       # complex64 type
    DT_MAX = 34             # Mark the boundaries of data types
```

调用示例如下：

```python
from torchair.ge import DataType
Cast(Tensor, dst_type=DataType.DT_BF16)
```
