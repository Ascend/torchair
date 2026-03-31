# readable\_cache

## 功能说明

开启[模型编译缓存功能](../../advanced/compile_cache.md)时需调用该接口读取封装的func函数缓存文件compiled\_module，并以可读文件格式（格式不限，如py、txt）呈现。

## 函数原型

```python
readable_cache(cache_bin, print_output=True, file=None)
```

## 参数说明

|**参数**|**输入/输出**|**说明**|
|--|--|--|
|cache_bin|输入|指定被封装func函数缓存文件的路径，例如：/home/workspace/.torchair_cache/Model_dynamic_f2df0818d06118d4a83a6cacf8dc6d28/prompt/compiled_module。|
|print_output|输入|是否打印func函数缓存文件解析后的内容。True：默认开启打印显示。False：不开启打印显示。|
|file|输入|解析生成的文件路径。默认为None，即不生成可读文件。绝对路径，如：/home/workspace/prompt.py。相对路径，如：prompt.py，默认在调用该接口的脚本所在工作目录下。|

## 返回值说明

返回文件内容，String类型。

## 约束说明

确保参数中指定的路径真实存在，并且运行用户具有读取和写入权限。

## 调用示例

参考[模型编译缓存功能\>使用方法](../../advanced/compile_cache.md#使用方法)。
