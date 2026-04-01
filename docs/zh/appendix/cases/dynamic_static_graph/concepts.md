# 动/静态图概念

## Dynamo中动/静态图概念

Dynamo编译后的FX图动/静态属性与模型输入的形状相关，因此先介绍torch.compile入参“**dynamic**”对模型输入形状的编译影响。

- **dynamic=False**：
    - 将tensor类型输入的具体维度值编译为固定常量（也称为固定形状），如\(10, 20\)。
    - 将scalar类型输入编译为固定常量（也称为固定值），如30。

- **dynamic=True**：
    - 将parameter/buffer类tensor输入的具体维度值编译为固定常量，如\(10, 20\)。
    - 将user\_input类tensor输入的具体维度值默认编译为符号（也称泛化为符号），如\(s0, s1\)；若此类tensor被mark\_static标记，其具体维度值被编译为固定常量，如\(10, 20\)。
    - 将scalar类型输入编译为符号，如s2。

**表 1**  不同类型输入编译规则

|dynamic参数配置|user_input tensor|mark_static的tensor|parameter/buffer tensor|scalar输入|
|--|--|--|--|--|
|dynamic=False|固定形状|固定形状|固定形状|固定值|
|dynamic=True|符号形状|固定形状|固定形状|符号|

dynamic=False时，编译后输入的所有维度具体化，可视为完全地静态编译；dynamic=True时，编译后输入默认符号化，且允许通过mark\_static局部静态化。

若Dynamo编译后FX图中的输入存在符号形状或符号scalar，通常称为FX动态shape图，否则称为FX静态shape图。

## GE动/静态图概念

GE提供了两种模型图调度的方式，Host调度和下沉调度，详细介绍参见《CANN 图模式开发指南》中“概念和原理介绍\>模型下沉调度”。

- Host调度：Host CPU将模型中的每个算子依次进行InferShape、Tiling计算、下发到Device执行操作。
- 下沉调度：在加载阶段，模型中的算子以整图形式提前下发到Device上。执行时，只需在Host侧下发一个模型执行的Task即可触发Device上模型调度执行。在下沉调度模型中，算子可以在加载阶段完成Tiling计算或静态Kernel编译，因此在执行态时无需再次计算Tiling。相比于Host调度模式，下沉调度模式可大大降低Host侧调度开销，有效减少Host和Device之间的交互，同时也可以解决Host调度的Host Bound。

GE动/静态shape图与调度模式：

- 对于所有输入tensor shape不固定的图，称为**动态shape图**。动态Shape图在执行时才能确定Shape，完成Tiling计算，只能采用Host调度。
- 对于所有输入tensor shape固定的图，称为**静态shape图**。静态shape图中的算子一般都能采用下沉调度。
- 对于部分输入tensor shape不固定，部分输入tensor shape固定的图，整图归类为**动态shape图**。静态部分可以采用下沉调度，动态部分可以采用Host调度。

    特殊情况下，静态shape图中存在值依赖算子，该算子默认使用Host调度。因此，在静态shape图中，部分算子采用下沉调度（能下沉的部分称为静态子图，在GE build图中以PartitionedCall节点形式存在），部分算子采用Host调度。

**表 1**  GE动静图shape特征表

|GE动/静态shape图|调度方式|
|--|--|
|动态shape图|所有算子采用Host调度。部分算子采用Host调度，部分算子采用下沉调度。|
|静态shape图|所有算子采用下沉调度。部分算子采用Host调度，部分算子采用下沉调度。|

通常，动态shape图也简称动态图，静态shape图也简称静态图。
