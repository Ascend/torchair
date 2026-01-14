# readable\_cache

## 功能说明

开启“**模型编译缓存功能**”时需要调用该接口读取封装的func函数缓存文件compiled\_module，并以可读文件格式（格式不限，如py、txt）呈现。

## 函数原型

```python
readable_cache(cache_bin, print_output=True, file=None)
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| cache_bin | 输入 | 指定被封装func函数缓存文件的路径。例如/home/workspace/.torchair_cache/Model_dynamic_f2df0818d06118d4a83a6cacf8dc6d28/prompt/compiled_module。 | 是 |
| print_output | 输入 | 是否打印func函数缓存文件解析后的内容。<br>  - True：默认开启打印。<br>  - False：不开启打印。 | 否 |
| file | 输出 | 解析生成的文件路径。默认为None，即不生成可读文件。<br>  - 绝对路径：如/home/workspace/prompt.py。<br>  - 相对路径：如prompt.py，默认在调用该接口的脚本所在工作目录下。 | 否 |

## 返回值说明

返回文件内容，String类型。

## 约束说明

请确保所有参数指定的路径真实存在，并且运行用户具有读、写操作权限。

## 调用示例

```python
import torch_npu, torchair
torchair.inference.readable_cache("/home/workspace/.torchair_cache/Model_dynamic_f2df0818d06118d4a83a6cacf8dc6d28/prompt/compiled_module", file="prompt.py")
```
