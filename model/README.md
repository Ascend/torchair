# TorchAir benchmarks

## 简介

为了评测图模式能力，pytorch社区在CI HUD中提供了图模式应用在torchbench/HuggingFace/Timm仓库的多个模型的加速效果。参考社区的实现，TorchAir提供了对torchbench仓和HuggingFace仓中部分模型的支持。用户可按照此README进行NPU图模式的测试。

## Torchbench

## 准备环境
1. 下载TorchAir源码
    ```
    git clone https://gitee.com/ascend/torchair.git
    ```

2. 切换到model目录中
    ```
    cd torchair/model
    ```

3. 下载pytorch/pytorch源码，复制其中的microbenchmarks目录至当前目录
    ```
    git clone https://github.com/pytorch/pytorch.git -b v2.1.0 --depth=1
    cp -r pytorch/benchmarks/dynamo/microbenchmarks .
    ```
如果下载失败，用户也可通过gitee的[同步仓](https://gitee.com/mirrors/pytorch)进行下载。

4. 下载pytorch/benchmark源码，并切换至指定commit id
    ```
    git clone  https://github.com/pytorch/benchmark.git --depth=1
    cd benchmark
    git remote set-branches origin '9910b31cc17d175a781412fd9ca6f18a4ee04610'
    git fetch --depth 1 origin 9910b31cc17d175a781412fd9ca6f18a4ee04610
    git checkout 9910b31cc17d175a781412fd9ca6f18a4ee04610
    cd ..
    ```

5. 在model目录中运行模型
    ```
    python3 torchbench.py --accuracy --cold-start-latency --train --amp --backend npu --only BERT_pytorch
    ```
    --accuracy指对图模式的精度进行验证。此外，用户可指定--performance来验证图模式的性能。

    --only参数指的是运行的模型，用户可参考[CI HUD](https://hud.pytorch.org/benchmark/torchbench/inductor_no_cudagraphs?startTime=Mon,%2011%20Mar%202024%2011:26:49%20GMT&stopTime=Mon,%2018%20Mar%202024%2011:26:49%20GMT&granularity=hour&mode=training&dtype=amp&lBranch=main&lCommit=c568b84794447b023747cd5a1bf47288fc657b77&rBranch=main&rCommit=660ec3d38d9d1c8567471ae7fe5b40ae7c6d7438)中的模型进行设置。