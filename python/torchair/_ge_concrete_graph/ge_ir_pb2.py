import pkg_resources
from torchair.core.utils import logger

# Import the corresponding protobuf Python interface based on the protobuf version in environment
protobuf_version = pkg_resources.get_distribution("protobuf").version

if pkg_resources.parse_version(protobuf_version) < pkg_resources.parse_version("4"):
    logger.debug('Generate protobuf Python interface for env with protobuf versions 3.x.')
    from torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2 import (ModelDef,
                                                                     GraphDef,
                                                                     OpDef,
                                                                     TensorDef,
                                                                     ShapeDef,
                                                                     AttrDef,
                                                                     TensorDescriptor,
                                                                     DataType)
else:
    logger.debug('Generate protobuf Python interface for env with protobuf versions 4.x.')
    from torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2 import (ModelDef,
                                                                     GraphDef,
                                                                     OpDef,
                                                                     TensorDef,
                                                                     ShapeDef,
                                                                     AttrDef,
                                                                     TensorDescriptor,
                                                                     DataType)
