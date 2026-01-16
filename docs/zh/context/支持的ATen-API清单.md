# 支持的ATen API清单

本章提供了支持入图的ATen API列表，这些API能力均对等Eager模式下的ATen API能力。

如果自定义模型用到的ATen API不在下表中，说明对应的API能力可能不完备，用户需根据实际情况进行Converter适配实现算子入图，具体步骤请参考[Converter补齐](https://gitcode.com/ascend/torchair/blob/master/CONTRIBUTING.md#converter%E8%A1%A5%E9%BD%90)。

**表 1**  ATen API清单


| ATen API名称 | 约束 |
| --- | --- |
| torch.nn.BatchNorm1d | 支持fp16，fp32<br>track_running_stats为True |
| torch.nn.BatchNorm2d | 支持fp16，fp32<br/>track_running_stats为True |
| torch.nn.BatchNorm3d | 支持fp16，fp32<br/>track_running_stats为True |
| torch._native_batch_norm_legit_no_training | 支持fp16，fp32 |
| torch._softmax | 支持fp32 |
| torch._softmax_backward_data | 支持fp32 |
| torch.ops.aten._to_copy.default | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.addmm | 支持fp16，fp32 |
| torch.bernoulli | 支持fp16，fp32 |
| torch.clone | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64 |
| torch.div | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.empty_like | 支持fp16，fp32，uint8，int8，int16，int32，int64 |
| torch.Tensor.expand | 支持bf16，fp16，fp32，int16，int32，int64 |
| torch.nn.functional.gelu | 支持bf16，fp16，fp32 |
| torch.nn.functional.hardswish | - 正向：支持fp16，fp32<br>- 反向：支持bf16，fp16，fp32，参数约束inplace=False |
| torch.nn.functional.leaky_relu | 支持bf16，fp16，fp32 |
| torch.mean | 支持bf16，fp16，fp32，complex64，complex128 |
| torch.native_layer_norm | 支持bf16，fp16，fp32 |
| torch.nn.functional.layer_norm | 支持bf16，fp16，fp32 |
| torch.ops.aten.new_empty_strided.default | - |
| torch.permute | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128 |
| torch.select | 支持fp32 |
| torch.ops.aten.slice.Tensor | 支持bf16，fp16，fp32 |
| torch.ops.aten.slice_backward.default | 支持bf16，fp16，fp32 |
| torch.sum | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.t | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64 |
| torch.nn.Threshold | 支持fp32，uint8，int8，int16，int32，int64 |
| torch.nn.MaxPool2d | - 正向：支持fp16，fp32，参数约束return_indice=False，dynamic=True<br>- 反向：支持fp16，fp32，bf16，input为四维 |
| torch.slice_scatter | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool |
| torch.nn.functional.avg_pool2d | - 正向：支持fp16，fp32，参数约束dynamic=False，kernel_size≤255，stride≤63<br>- 反向：支持fp16，fp32 |
| torch.nn.functional.group_norm | 支持bf16，fp16，fp32 |
| sigmoid | 支持fp16，fp32，int8，int32，int64 |
| torch.sigmoid | 支持bf16，fp16，fp32，fp64 |
| torch._native_batch_norm_legit | 支持fp16，fp32<br>参数约束track_running_stats=False |
| torch.index_select | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool |
| torch.sub | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64 |
| torch.fill | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128 |
| self.new_zeros(...) | 支持fp16，fp32，fp64，bf16，int8，int16，int32，int64，uint8，bool，complex64，complex128 |
| torch.mul | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128 |
| torch.nn.functional.hardsigmoid | 支持fp16，fp32 |
| torch.clamp | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64<br>参数约束min<max |
| torch.ops.aten.alias.default | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool |
| self.new_ones(...) | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool |
| torch.nn.functional.embedding | 支持int32，int64<br>参数约束max_norm=False，sparse=False，scale_grad_by_freq=False前向不支持，_freeze=False |
| torch.nn.functional.sigmoid | 支持bf16，fp16，fp32，fp64 |
| torch.nn.functional.hardtanh | - 正向：<br>   当min_val和max_val为浮点型时，input支持bf16，fp16，fp32，fp64<br>   当min_val和max_val为整型时，input支持int32，int64<br>- 反向：支持bf16，fp16，fp32 |
| torch.stack | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64 |
| torch.unbind | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool |
| torch.nn.functional.pad | 支持bf16，fp16，fp32，int16，int32，int64 |
| torch.constant_pad_nd | 支持bf16，fp16，fp32，int32，int64，bool<br>参数约束inplace=False |
| torch.nn.avgpool2d | 支持fp16，fp32<br>参数约束dynamic=False，kernelsize≤255，stride≤63 |
| torch.nn.Conv2d | 支持fp16，fp32 |
| torch.nn.Conv1d | 支持fp16，fp32 |
| torch.nn.ConvTranspose1d | 支持fp16，fp32 |
| torch.nn.ConvTranspose2d | 支持fp16，fp32 |
| torch.ops.aten.floor.default | 支持fp16，fp32 |
| torch.nn.AdaptiveAvgPool2d | 支持fp16，fp32，output_size仅支持list |
| torch.fft.fft | 支持complex64 |
| torch.fft.fft2 | 支持complex64 |
| torch.nn.LogSoftmax | 支持bf16，fp16，fp32 |
| torch.abs | 支持fp16，fp32，uint8，int8，int16，int32，int64 |
| torch.arange | 支持bf16，fp16，fp32，int32，int64 |
| torch.argmax | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64 |
| torch.as_strided | 支持fp32 |
| torch.as_strided_scatter | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128<br>输入tensor src的stride属性要小于stride参数 |
| torch.bitwise_and | 支持uint8，int8，int16，int32，int64，bool |
| torch.bitwise_not | 支持uint8，int8，int16，int32，int64，bool |
| torch.ceil | 支持fp32，fp16 |
| torch.cumsum | 支持bf16，fp16，fp32，int8，int16，int32，int64，bool |
| torch.detach | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.nn.functional.elu | 支持fp16，fp32 |
| torch.empty | 支持fp16，fp32，uint8，int8，int16，int32，int64 |
| torch.exp | 支持bf16，fp16，fp32，int64，bool，complex64，complex128 |
| torch.nn.functional.grid_sample | 支持fp16，fp32，fp64 |
| torch.ops.aten.index.Tensor | 支持fp16，fp32，uint8，int8，int32，int64，bool |
| torch.index_add | 支持fp16，fp32，int64，bool |
| torch.le | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.linspace | 参数start和end为浮点类型，参数steps为非负整数 |
| torch.log | 支持bf16，fp16，fp32，int64，bool，complex64，complex128 |
| torch.logsumexp | 支持fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.max | 支持bf16，fp16，fp32，int64，bool |
| torch.minimum | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.nn.functional.mse_loss | - 正向：支持fp16，fp32<br>- 负向：支持fp16，fp32，参数self和target的数据类型要一致 |
| torch.new_empty | 支持fp32 |
| torch.new_full | 支持int64 |
| torch.nn.functional.nll_loss | - 正向：<br>  - 参数weight和input的数据类型必须相同，数据类型支持bf16，fp16，fp32<br>  - 参数target的数据类型支持uint8，int32，int64<br>  - 当参数self为1D-tensor时，target需要为0D-tensor；当self为2D-tensor时，target需要为1D-tensor，两个参数的shape关系需满足：self.shape[0] = target.shape[0]<br>- 反向：支持bf16，fp16，fp32，int64 |
| torch.ones_like | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.rand | 输入为生成tensor的shape |
| torch.randint | 输入为生成tensor的shape |
| torch.randn | 输入为生成tensor的shape |
| torch.remainder | 支持fp16，fp32，int32，int64 |
| torch.scatter | 支持int16，int32，int64 |
| torch.scatter_add | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool<br>参数index必须为非空tensor |
| torch.select_scatter | 支持fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.sgn | 支持bf16，fp16，fp32，int32 |
| torch.sort | 支持fp16，fp32，uint8，int8，int16，int32，int64 |
| torch.sqrt | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128 |
| torch.nn.functional.tanh | 支持fp16，fp32，bf16 |
| torch.triu | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool |
| torch.view_as_complex | 支持fp32，fp64 |
| torch.view_as_real | 支持complex64 |
| torch.where | 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool<br>输入tensor的shape小于8维<br>不支持动态场景torch.where(condition) → tuple of LongTensor |
| torch.zeros | 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64 |
| torch.zeros_like | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128 |
| torch.eq | 支持bf16，fp16，fp32，uint8，int8，int32，int64，bool |
| torch.erf | 支持bf16，fp16，fp32， |
| torch.full_like | 支持fp16，fp32，int32，uint8，int16，int64，bool，complex64，complex128 |
| torch.lt | 支持bf16，fp16，fp32，uint8，int8，int32，int64 |
| torch.ne | 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128 |
| torch.nn.functional.glu | - 正向：支持fp16，fp32，dim值在input维度范围内，且input.shape[dim]为偶数<br>- 反向：支持fp16，fp32 |
| torch.ge | 输入Tensor支持bool，uint8，int8，int16，int32，int64，fp16，bf16，fp32，fp64 |
| torch.ops.aten.sym_size.int | 输入Tensor支持bf16，fp32，fp16，int8，uint8，int16，int32，int64 |
| math.ceil | 支持bf16，fp32，fp16，fp64 |
| torch.logical_and | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128 |
| torch._foreach_abs | 支持bf16，fp16，fp32 |
| torch._foreach_acos | 支持bf16，fp16，fp32 |
| torch._foreach_add | 支持bf16，fp16，fp32，int32<br>- tensor是bf16，fp16，fp32时，scalar支持fp32<br>- tensor是int32，scalar支持int64 |
| torch._foreach_addcdiv | 支持fp32，bf16 |
| torch._foreach_addcmul | 支持fp32，bf16 |
| torch._foreach_asin | 支持bf16，fp16，fp32 |
| torch._foreach_atan | 支持bf16，fp16，fp32 |
| torch._foreach_ceil | 支持bf16，fp16，fp32 |
| torch._foreach_cos | 支持bf16，fp16，fp32 |
| torch._foreach_cosh | 支持bf16，fp16，fp32 |
| torch._foreach_div | 支持bf16，fp16，fp32，int32<br>- tensor是bf16，fp16，fp32，scalar支持fp32<br>- tensor是int32，scalar支持int64 |
| torch._foreach_erf | 支持bf16，fp16，fp32 |
| torch._foreach_erfc | 支持bf16，fp16，fp32 |
| torch._foreach_exp | 支持bf16，fp16，fp32 |
| torch._foreach_expm1 | 支持bf16，fp16，fp32 |
| torch._foreach_floor | 支持bf16，fp16，fp32，int8，uint8，int32，int64 |
| torch._foreach_frac | 支持bf16，fp16，fp32，int8，uint8，int32，int64 |
| torch._foreach_lerp | 支持fp16，fp32，bf16 |
| torch._foreach_log | 支持bf16，fp16，fp32 |
| torch._foreach_log10 | 支持bf16，fp16，fp32 |
| torch._foreach_log1p | 支持bf16，fp16，fp32 |
| torch._foreach_log2 | 支持bf16，fp16，fp32 |
| torch._foreach_maximum | 支持fp16，fp32，int32，bf16 |
| torch._foreach_minimum | 支持fp16，fp32，int32，bf16 |
| torch._foreach_mul | 支持bf16，fp16，fp32，int32<br>- tensor是bf16，fp16，fp32，scalar支持fp32<br>- tensor是int32，scalar支持int64 |
| torch._foreach_norm | 支持fp16，fp32，bf16 |
| torch._foreach_pow | 支持bf16，fp16，fp32<br>- tensor_list输入dtype和scalar dtype相同<br>- tensor_list输入dtype为浮点数，scalar type为int |
| torch._foreach_round | 支持bf16，fp16，fp32，int32，int64 |
| torch._foreach_sigmoid | 支持bf16，fp16，fp32 |
| torch._foreach_sin | 支持bf16，fp16，fp32 |
| torch._foreach_sinh | 支持bf16，fp16，fp32 |
| torch._foreach_sub | 支持bf16，fp16，fp32，int32<br>- tensor_list输入dtype和scalar dtype相同<br>- tensor_list输入dtype为浮点数，scalar type为int |
| torch._foreach_tan | 支持bf16，fp16，fp32 |
| torch._foreach_tanh | 支持bf16，fp16，fp32 |
| torch._foreach_trunc | 支持bf16，fp16，fp32 |
| torch.searchsorted | 支持fp16，fp32，uint8，int8，int16，int32，int64 |
| torch.sym_sum | 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128 |

