# set\_dim\_gears

## 功能说明

开启[动态shape图分档执行功能](动态shape图分档执行功能.md)时需要调用该接口设置图被划分的档位。

## 函数原型

```python
set_dim_gears(t: torch.Tensor, dim_gears: Dict[int, Union[List[int], Tuple[int]]])
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| t | 输入 | 待分档的输入Tensor。 | 是 |
| dim_gears | 输入 | 用于设置Tensor不同dim维度下的档位值。输入类型为Dict（{key:value}键值对形式），其中key为dim维度（整型），value为档位值（整数列表或元组）。 | 是 |

## 返回值说明

无

## 约束说明

- 本功能仅支持max-autotune模式，暂不支持同时配置[Dynamo导图功能](Dynamo导图功能.md)、[RefData类型转换功能](RefData类型转换功能.md)。

- 本功能仅适用于整图优化场景。

- [set\_dim\_gears](set_dim_gears.md)需和torch.compile中的dynamic=True搭配使用。因为set\_dim\_gears只会符号化入参指定的Tensor及维度，其他scalar值的符号化会在dynamic=True时由Dynamo自动完成。

- 本功能要求网络中参与分档的Tensor不能传入私有格式，如FRACTAL\_NZ、NC1HWC0等。

-  dim_gears使用说明： 

    - 支持对同一个Tensor设置一个或者多个维度的档位。
    -  若图编译、执行时Tensor的shape不在设置的档位中，会导致编译或执行报错，请合理设置档位值。 
    -  不支持对同一个Tensor使用该接口设置两次不一样的档位。 
    -  生成的总档位数量不超过100，档位值不能包含0或1，因为动态FX graph中dim值符号化的最大表示范围是[2, ∞)，因此当dim为0或1时，不会命中动态的FX graph，需要重新成图，因此无法执行分档流程。 
    -  首次执行时对输入Tensor设置档位即可，保证首次编译时能够获取到档位，后续执行时无需设置档位，避免因为设置档位的动作引发性能下降。

## 调用示例

```python
import torch, torch_npu, torchair
input1 = torch.ones(2, 2).npu()
torchair.inference.set_dim_gears(input1, dim_gears={0:[2, 4]})
```
