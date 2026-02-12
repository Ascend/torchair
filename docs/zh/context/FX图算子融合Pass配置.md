# FX图算子融合Pass配置

## 功能简介

图模式下，TorchAir集成了PyTorch原生Pattern能力的算子融合功能，能够通过特定的算子替换规则，使用融合算子替换FX图中多个算子。这种优化可以有效减少部分场景下不必要的下发开销，提高模型执行效率。当与其它图优化策略结合使用时，可通过优化对比来选择最佳方案。

目前TorchAir提供了多种**默认的算子融合Pass**（适用于Deepseek、Long-Cat等网络），参见[表1](#table1)，符合替换规则的算子组合可被替换成对应的融合算子。

**表 1**  已支持的算子融合Pass  <a name="table1"></a>


| 替换规则 | 对应的融合算子 |
| --- | --- |
| npu_add_rms_norm输出直接作为npu_dynamic_quant（含smooth_scales参数）输入 | npu_add_rms_norm_dynamic_quant |
| npu_add_rms_norm输出经flatten(0,1) 后作为npu_dynamic_quant（不含smooth_scales参数）输入，且npu_dynamic_quant输出的scaleOut执行view(-1,1) | npu_add_rms_norm_dynamic_quant（并处理flatten与view操作） |
| npu_add_rms_norm输出先获取最后一维尺寸h，再经view(-1, h)变形及to(torch.float32)类型转换 | npu_add_rms_norm_cast（并处理view） |
| matmul输出作为transpose输入，transpose参数仅支持(0,1)或者(1,0) | npu_transpose_batchmatmul |
| npu_add_rms_norm输出作为npu_quantize输入，npu_add_rms_norm输入尾轴需32B对齐，并满足融合算子npu_add_rms_norm_quant约束条件 | npu_add_rms_norm_quant |

各融合Pass的算子替换逻辑参见[融合规则](#融合规则)，融合算子的输出被正常使用，且融合后不再存在的中间结果不得被其他位置使用，否则无法融合。另外，用户可通过[register\_replacement](register_replacement.md)接口**实现自定义算子融合Pass注册**（参见接口调用示例），注意需自行保证融合规则的正确性。

## 使用约束

-   本功能依赖PyTorch 2.6.0或更高版本。
-   无论是默认支持的算子融合Pass还是自定义的算子融合Pass，均可由pattern\_fusion\_pass配置。
-   表1中matmul输入必须是三维，npu_transpose_batchmatmul及npu_add_rms_norm_quant算子融合max-autotune模式不生效。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，**默认开启**，关闭的示例如下，仅供参考不支持直接拷贝运行，参数介绍参见[表2](#table2)。

```python
import torch_npu
import torchair
config = torchair.CompilerConfig()
# FX图中算子融合Pass配置
config.experimental_config.pattern_fusion_pass = False
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 2**  参数说明 <a name="table2"></a>


| 参数名 | 说明 |
| --- | --- |
| pattern_fusion_pass | FX图是否开启算子融合Pass配置，布尔类型。<br>- False：关闭。<br>- True（默认值）：开启。 |

设置成功后，参考[图结构dump功能](图结构dump功能.md)开启FX图dump，假设原始FX图满足npu\_add\_rms\_norm\_dynamic\_quant的替换规则，可从图结构中看到如下类似的信息，打印信息表明已经存在对应融合算子。

```
# No stacktrace found for following nodes
npu_add_rms_norm_dynamic_quant_default = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(arg2_1, arg1_1, arg0_1, output_mask = [True, True]);  arg2_1 = arg1_1 = arg0_1 = None
getitem_5: "i8[2, 3, 4]" = npu_add_rms_norm_dynamic_quant_default[0]
getitem_6: "f16[2, 3, 4]" = npu_add_rms_norm_dynamic_quant_default[2]
getitem_7: "f32[2, 3]" = npu_add_rms_norm_dynamic_quant_default[3];  npu_add_rms_norm_dynamic_quant_default = None
view_default: "i8[6, 4]" = torch.ops.aten.reshape.default(getitem_5, [6, 4]);  getitem_5 = None
view_default_1: "f32[6, 1]" = torch.ops.aten.reshape.default(getitem_7, [-1, 1]);  getitem_7 = None
return (view_default, view_default_1, getitem_6)
```

## 融合规则
-   npu_add_rms_norm_dynamic_quant
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#0066cc', 'lineColor': '#000000'}}}%%

flowchart LR
    subgraph Before["融合前"]
        direction TB
        X1["x1"] & X2["x2"] & Gamma["gamma"] --> AddRMS["npu_add_rms_norm"]
        AddRMS --> y["y"]
        y["y"] --> DynQuant["npu_dynamic_quant"]
        Smooth["smooth_scales"] --> DynQuant
        DynQuant --> YOut["yOut"] & Scale1["scale1Out"]
        AddRMS --> XOut["xOut"]
    end

    subgraph After["融合后"]
        direction TB
        X1_2["x1"] & X2_2["x2"] & Gamma2["gamma"] & Smooth2["smooth_scale"] --> Fused["npu_add_rms_norm_dynamic_quant"]
        Fused --> YOut2["yOut"] & Scale2["scale1Out"] & XOut2["xOut"]
    end

    Before ==> After

    linkStyle default stroke:#000000,stroke-width:2px,fill:none

    style Before fill:#ffffff,stroke:none
    style After fill:#ffffff,stroke:none
    style X1 fill:#cce0ff,stroke:none
    style X2 fill:#cce0ff,stroke:none
    style Gamma fill:#cce0ff,stroke:none
    style y fill:#cce0ff,stroke:none
    style Smooth fill:#cce0ff,stroke:none
    style AddRMS fill:#cce0ff,stroke:none
    style DynQuant fill:#cce0ff,stroke:none
    style YOut fill:#cce0ff,stroke:none
    style Scale1 fill:#cce0ff,stroke:none
    style XOut fill:#cce0ff,stroke:none
    style X1_2 fill:#cce0ff,stroke:none
    style X2_2 fill:#cce0ff,stroke:none
    style Gamma2 fill:#cce0ff,stroke:none
    style Smooth2 fill:#cce0ff,stroke:none
    style Fused fill:#cce0ff,stroke:none
    style YOut2 fill:#cce0ff,stroke:none
    style Scale2 fill:#cce0ff,stroke:none
    style XOut2 fill:#cce0ff,stroke:none

```

-   npu_add_rms_norm_dynamic_quant（并处理flatten与view操作）
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#0066cc', 'lineColor': '#000000'}}}%%

flowchart LR
    subgraph Before2["融合前"]
        direction TB
        X1_3["x1"] & X2_3["x2"] & Gamma3["gamma"] --> AddRMS2["npu_add_rms_norm"]
        AddRMS2 --> ViewY["y.flatten(0,1)"] --> DynQuant2["npu_dynamic_quant"]
        DynQuant2 --> YOut3["yOut"] & ScaleRaw["scale1Out"]
        ScaleRaw --> ViewScale["View(-1,1)"]
        AddRMS2 --> XOut3["xOut"]
    end

    subgraph After2["融合后"]
        direction TB
        X1_4["x1"] & X2_4["x2"] & Gamma4["gamma"] --> Fused2["npu_add_rms_norm_dynamic_quant"]
        Fused2 --> YOut4["yOut"] --> FlattenAuto["flatten(0,1)"]
        Fused2 --> Scale3["scale1Out"] --> ViewAuto["view(-1,1)"]
        Fused2 --> XOut4["xOut"]
    end

    Before2 ==> After2

    linkStyle default stroke:#000000,stroke-width:2px,fill:none

    style Before2 fill:#ffffff,stroke:none
    style After2 fill:#ffffff,stroke:none
    style X1_3 fill:#cce0ff,stroke:none
    style X2_3 fill:#cce0ff,stroke:none
    style Gamma3 fill:#cce0ff,stroke:none
    style AddRMS2 fill:#cce0ff,stroke:none
    style ViewY fill:#cce0ff,stroke:none
    style DynQuant2 fill:#cce0ff,stroke:none
    style YOut3 fill:#cce0ff,stroke:none
    style ScaleRaw fill:#cce0ff,stroke:none
    style ViewScale fill:#cce0ff,stroke:none
    style XOut3 fill:#cce0ff,stroke:none
    style X1_4 fill:#cce0ff,stroke:none
    style X2_4 fill:#cce0ff,stroke:none
    style Gamma4 fill:#cce0ff,stroke:none
    style Fused2 fill:#cce0ff,stroke:none
    style YOut4 fill:#cce0ff,stroke:none
    style FlattenAuto fill:#cce0ff,stroke:none
    style Scale3 fill:#cce0ff,stroke:none
    style ViewAuto fill:#cce0ff,stroke:none
    style XOut4 fill:#cce0ff,stroke:none
```
-   npu_add_rms_norm_cast（并处理view） 

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#0066cc', 'lineColor': '#000000'}}}%%

flowchart LR
    subgraph Before["融合前"]
        direction TB
        X1["x1"] & X2["x2"] & Gamma["gamma"] --> AddRMS["npu_add_rms_norm"]
        AddRMS --> Y["y"]
        AddRMS --> XOut["xOut"]
        Y --> Size["size(-1)"] --> H["h"]
        Y --> View["View(-1, h)"] --> Cast["_npu_dtype_cast<br/>to float32"]
    end

    subgraph After["融合后"]
        direction TB
        X1_2["x1"] & X2_2["x2"] & Gamma2["gamma"] --> Fused["npu_add_rms_norm_cast"]
        Fused --> Y2["y"]
        Fused --> XOut2["xOut"]
        Fused --> YCast["y_cast"]
        YCast --> View2["View(-1, h)"]
    end

    Before ==> After

    linkStyle default stroke:#000000,stroke-width:2px,fill:none

    style Before fill:#ffffff,stroke:none
    style After fill:#ffffff,stroke:none
    style X1 fill:#cce0ff,stroke:none
    style X2 fill:#cce0ff,stroke:none
    style Gamma fill:#cce0ff,stroke:none
    style AddRMS fill:#cce0ff,stroke:none
    style Y fill:#cce0ff,stroke:none
    style XOut fill:#cce0ff,stroke:none
    style Size fill:#cce0ff,stroke:none
    style H fill:#cce0ff,stroke:none
    style View fill:#cce0ff,stroke:none
    style Cast fill:#cce0ff,stroke:none
    style X1_2 fill:#cce0ff,stroke:none
    style X2_2 fill:#cce0ff,stroke:none
    style Gamma2 fill:#cce0ff,stroke:none
    style Fused fill:#cce0ff,stroke:none
    style Y2 fill:#cce0ff,stroke:none
    style XOut2 fill:#cce0ff,stroke:none
    style YCast fill:#cce0ff,stroke:none
    style View2 fill:#cce0ff,stroke:none
```
-   npu_transpose_batchmatmul
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#0066cc', 'lineColor': '#000000'}}}%%

flowchart LR
    subgraph Before["融合前"]
        direction TB
        X1["x1"] & X2["x2"]  --> batchmatmul["matmul"]
        batchmatmul --> transpose["transpose"]
        transpose["transpose"]--> Out["Out"]
    end

    subgraph After["融合后"]
        direction TB
        X1_2["x1"] & X2_2["x2"] --> Fused["npu_transpose_batchmatmul"]
        Fused --> out["Out"]
    end

    Before ==> After

    linkStyle default stroke:#000000,stroke-width:2px,fill:none

    style Before fill:#ffffff,stroke:none
    style After fill:#ffffff,stroke:none
    style X1 fill:#cce0ff,stroke:none
    style X2 fill:#cce0ff,stroke:none
    style batchmatmul fill:#cce0ff,stroke:none
    style transpose fill:#cce0ff,stroke:none
    style Out fill:#cce0ff,stroke:none
    style Fused fill:#cce0ff,stroke:none
    style out fill:#cce0ff,stroke:none
    style X1_2 fill:#cce0ff,stroke:none
    style X2_2 fill:#cce0ff,stroke:none

```
-   npu_add_rms_norm_quant
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#0066cc', 'lineColor': '#000000'}}}%%

flowchart LR
    subgraph Before["融合前"]
        direction TB
        X1["x1"] & X2["x2"] & Gamma["gamma"] --> AddRMS["npu_add_rms_norm"]
        AddRMS --> y["y"]
        y["y"] --> Quant["npu_quantize"]
        Scales["scales"] --> Quant
	ZeroPoints["zero_points"] --> Quant
        Quant --> YOut["yOut"] 
        AddRMS --> XOut["xOut"]
    end

    subgraph After["融合后"]
        direction TB
        X1_2["x1"] & X2_2["x2"] & Gamma2["gamma"] & Scales2["scales"] & ZeroPoints2["zero_points"] --> Fused["npu_add_rms_norm_quant"]
        Fused --> YOut2["yOut"]  & XOut2["xOut"]
    end

    Before ==> After

    linkStyle default stroke:#000000,stroke-width:2px,fill:none

    style Before fill:#ffffff,stroke:none
    style After fill:#ffffff,stroke:none
    style X1 fill:#cce0ff,stroke:none
    style X2 fill:#cce0ff,stroke:none
    style Gamma fill:#cce0ff,stroke:none
    style y fill:#cce0ff,stroke:none
    style Scales fill:#cce0ff,stroke:none
    style AddRMS fill:#cce0ff,stroke:none
    style Quant fill:#cce0ff,stroke:none
    style YOut fill:#cce0ff,stroke:none
    style ZeroPoints fill:#cce0ff,stroke:none
    style XOut fill:#cce0ff,stroke:none
    style X1_2 fill:#cce0ff,stroke:none
    style X2_2 fill:#cce0ff,stroke:none
    style Gamma2 fill:#cce0ff,stroke:none
    style Scales2 fill:#cce0ff,stroke:none
    style ZeroPoints2 fill:#cce0ff,stroke:none
    style Fused fill:#cce0ff,stroke:none
    style YOut2 fill:#cce0ff,stroke:none
    style XOut2 fill:#cce0ff,stroke:none
```
