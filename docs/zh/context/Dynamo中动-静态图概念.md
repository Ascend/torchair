# Dynamo中动/静态图概念

Dynamo编译后的FX图动/静态属性与模型输入的形状相关，因此先介绍torch.compile入参“**dynamic**”对模型输入形状的编译影响。

-   **dynamic=False**：
    -   将tensor类型输入的具体维度值编译为固定常量（也称为固定形状），如\(10, 20\)。
    -   将scalar类型输入编译为固定常量（也称为固定值），如30。

-   **dynamic=True**：
    -   将parameter/buffer类tensor输入的具体维度值编译为固定常量，如\(10, 20\)。
    -   将user\_input类tensor输入的具体维度值默认编译为符号（也称泛化为符号），如\(s0, s1\)；若此类tensor被mark\_static标记，其具体维度值被编译为固定常量，如\(10, 20\)。
    -   将scalar类型输入编译为符号，如s2。

**表 1**  不同类型输入编译规则


| dynamic参数配置 | user_input tensor | mark_static的tensor | parameter/buffer tensor | scalar输入 |
| --- | --- | --- | --- | --- |
| dynamic=False | 固定形状 | 固定形状 | 固定形状 | 固定值 |
| dynamic=True | 符号形状 | 固定形状 | 固定形状 | 符号 |

dynamic=False时，编译后输入的所有维度具体化，可视为完全地静态编译；dynamic=True时，编译后输入默认符号化，且允许通过mark\_static局部静态化。

若Dynamo编译后FX图中的输入存在符号形状或符号scalar，通常称为FX动态shape图，否则称为FX静态shape图。

