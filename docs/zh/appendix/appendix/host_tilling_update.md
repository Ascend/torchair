# npugraph\_ex场景下Host Tilling参数刷新功能介绍

## 功能简介

aclgraph支持在Replay前刷新内核（kernels）功能，能够在类FIA算子执行前，根据实际计算出的actual\_seq\_len值动态更新分块（Tiling）结果，从而确保此类算子能够被aclgraph成功捕获并消除Tiling计算带来的调度间隙。

如下图所示，主要通过以下两个阶段实现：

1. Capture阶段：npugraph\_ex通过aclgraph update接口，在Capture阶段记录需要更新参数的算子，并插入event wait，控制更新时的时序。
2. Replay阶段：npugraph\_ex调用aclgraph replay接口后，下发需要更新参数的算子，并插入event record，保证参数更新后，算子执行。

![](../../figures/ScreenShot_20260209203104.png)

## 使用方法

参见[npugraph\_ex快速上手](../../npugraph_ex/quick_start.md)提供的使用方法。
