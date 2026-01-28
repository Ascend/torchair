# 算子融合规则配置功能（optimization\_switch）

## 功能简介

算子编译时，可根据实际业务需要灵活地配置算子融合，以降低网络推理时间、提高整网性能。

本功能是算子融合规则的控制开关，与[算子融合规则配置功能（fusion\_switch\_file）](算子融合规则配置功能（fusion_switch_file）.md)类似，差异如下：

config.fusion\_config.fusion\_switch\_file仅能关闭图融合和UB（Unified Buffer）融合的规则，并且需要单独配置Json文件；而本功能适用于所有融合规则的指定，不需要再单独设置Json文件。如果两个功能都配置，且配置了同一个融合规则，则以本功能配置为准。

## 使用约束

本功能仅支持max-autotune模式。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数说明如下表。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# 算子融合规则配置开关
config.ge_config.optimization_switch = "Passname1:on;Passname2:off"
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend) 
```

**表 1**  参数说明


| 参数名 | 说明 |
| --- | --- |
| optimization_switch | 算子编译时，融合规则的控制开关。取值格式为key-value键值对，形如"Passname1:on;Passname2:off"，key为Pass名称，value为on（表示开）或off（表示关），不支持大小写模式匹配，多组配置使用英文分号分隔。可配置的融合规则请参见[融合规则列表](#融合规则列表)。 |

## 融合规则列表

>**须知：** 注意如下融合规则关闭后，可能会对功能使用有影响，请谨慎操作。更多融合规则的介绍请参见[《CANN 图融合和UB融合规则参考》](https://hiascend.com/document/redirect/CannCommunitygraphubfusionref)。

-   AABiasaddConvFusion
-   AddNFusionPass
-   AddRmsNormFusionGraphPass
-   ADeformableConv2dPass
-   ADepthwiseFusionPass
-   ALSTMFusionPass
-   ApplyAddOutputPass
-   AReduceMeanFusionPass
-   AReduceSumFusionPass
-   ArgMaxWithFusionPass
-   AvgPool3DFusionPass
-   AvgPool3DGradFusionPass
-   AvgPoolGradFusionPass
-   AvgPoolQuantProcessFusionPass
-   BatchMatMulFusionPass
-   BatchMatmulV2QuantProcessFusionPass
-   BatchNormBnInferFusionPass
-   BatchNormGradBnInferGradFusion
-   BatchNormGradInfGradFusion
-   CastRemoveFusionPass
-   clip\_by\_norm\_nodivsquaresum
-   CommonLSTMFusionPass
-   CommonSubexpressionEliminationPass：GE公共表达式消除Pass
-   ConstToAttrGatherV2Fusion
-   ConstToAttrPass
-   ConstToAttrReduceSumFusion
-   ConstToAttrResizeNearestNeighborGradFusion
-   ConstToAttrStridedSliceV2Fusion
-   Conv2DQuantProcessFusionPass
-   Conv2DTDQuantProcessFusionPass
-   COPYPass
-   DeConvQuantProcessFusionPass
-   DeformableOffsetsFusionPass
-   DepthwiseDfFusionPass
-   DepthwiseDwMulFusionPass
-   DepthwiseFusionPass
-   DepthwiseInsertTransDataFusionPass
-   DepthwiseToConv2dFusionPass
-   DreluFusionPass
-   DWConv2DQuantProcessFusionPass
-   DynamicGRUV2GradFusionPass
-   DynamicRNNFusionPass
-   DynamicRNNGradAFusionPass
-   DynamicRNNGradAlignFusionPass
-   DynamicRNNGradDAlignFusionPass
-   DynamicRNNGradDFusionPass
-   DynamicRNNGradFusionPass
-   DynamicRNNInsertTransposePass
-   DynamicRNNInsertTransposePass
-   DynamicRNNSeqFusionPass
-   EinsumPass
-   FCQuantProcessFusionPass
-   FixPipeAbilityProcessPass
-   FlattenV2Pass
-   FusedBatchNormBertFusionPass
-   FusedBatchnormFusionPass
-   FusedBatchNormGradFusionPass
-   FusedBatchNormGradFusionPass
-   GlobalAvgPoolPass
-   GroupConv2DQuantProcessFusionPass
-   HostBNFusionPass
-   HostShapeOptimizationPass：AI CPU断流水pass
-   In-placeAddRmsNormFusionPass
-   MapIndexFusionPass
-   MatMul2MatMulV2FusionPass
-   MatmulV2QuantProcessFusionPass
-   MaxPoolWithArgmaxFusionPass
-   NormalizeFusionPass
-   PackFusionPass
-   PassThroughFusionPass
-   PassThroughSecondFusionPass
-   PermuteFusionPass
-   PoolingQuantProcessFusionPass
-   PReluGradFusionPass
-   PriorBoxPass
-   ProposalFusionPass
-   RNNFusionPass
-   sedBatchNormGradFusionPass
-   SingleBatchNormFusion
-   SoftmaxGradExtFusion
-   SpatialTransformerDPass
-   SPPPass
-   TbeAntiquantMaxpoolingFusionPass
-   TbeConvCommonRules0FusionPass
-   TbeConvCommonRules2FusionPass
-   TbeConvDequantS16FusionPass
-   TbeConvRequantFusionPass
-   TfMergeConv2DBackpropInputFusionPass
-   TfMergeSubFusionPass
-   TfTagNoConstFoldingFusionPass
-   TransdataCastFusionPass
-   TransposedUpdateFusionPass
-   UnpackFusionPass
-   V100NotRequantFusionPass
-   V200NotRequantFusionPass
-   WeightQuantBatchMatmulV2TransposeFusionPass
-   YoloPass
-   YoloV2DetectionOutputPass
-   YoloV3DetectionOutputV2Pass
-   ZConcatDFusionPass
-   ZConcatExt2FusionPass
-   ZConfusionSoftmaxGradFusionPass
-   ZSplitVDFusionPass
-   ZSplitVFusionPass

