class ConverterErrorMsg:
    _AUTO_GENERATED_CONVERTER_HEADER = "Failed to auto-generated converter for {target} to AscendIR: "
    _ARGS_CONVERSION_HEADER = "Failed to convert torch args to AscendIR args: "
    _CUSTOM_OP_GUIDE_HINT = (
        "If necessary, you can check the Ascend Extension for Pytorch version documentation "
        "and refer to the custom op guide to implement your own converter."
    )
    _MANUAL_IMPLEMENTATION_SOLUTION = (
        "You have two methods to handle this issue: "
        "1. Modify the torch or AscendIR parameters to match them. "
        "2. Implement your own converter based on the Ascend Extension for Pytorch version documentation."
    )
    _PARAMS_DEFINITION_CHECK_HINT = (
        "If implementing the converter by yourself, please check if the params of the torchair.ge.custom_op(...) match the AscendIR definition. "
        "If your converter is auto-generated, please check if the params of the torch match the AscendIR definition."
    )

    # 不支持OpOverload实例
    NOT_OP_OVERLOAD = (
        _AUTO_GENERATED_CONVERTER_HEADER +
        "{target} is not an instance of OpOverload and does not support auto-generate converter."
    )

    # 不支持torch内置算子
    BUILTIN_OP = (
        _AUTO_GENERATED_CONVERTER_HEADER +
        "PyTorch built-in ops not support auto-generate converters. " +
        _CUSTOM_OP_GUIDE_HINT
    )

    # 不支持scalar参数
    SCALAR_INPUT = (
        _AUTO_GENERATED_CONVERTER_HEADER +
        "this op has scalar input: {name}, not support auto-generate converter. " +
        _CUSTOM_OP_GUIDE_HINT
    )

    # SO文件加载失败
    SO_LOAD_FAILED = (
        "Failed to load GetRegisteredIrDef from libge_runner.so. "
        "Please make sure that CANN is installed correctly and that the CANN environment variables have been successfully loaded."
    )

    # IR未注册
    GE_IR_NOT_REGISTERED = (
        "No AscendIR {name} was found to be registered. Please make sure the custom op is successfully registered, "
        "If you need to view logs to assist in positioning, you can set the environment variable ASCEND_GLOBAL_LOG_LEVEL and ASCEND_SLOG_PRINT_TO_STDOUT view op registration related logs. "
        "If necessary, you can refer to the CANN version documentation to view the description of environment variables."
    )

    # tensor参数数量不匹配
    TENSOR_INPUTS_COUNT_MISMATCH = (
        _AUTO_GENERATED_CONVERTER_HEADER +
        "the number of torch tensor inputs does not match the AscendIR {ge_name} inputs. torch tensor inputs: {tensor_inputs}, AscendIR inputs: {ge_inputs}. " +
        _MANUAL_IMPLEMENTATION_SOLUTION
    )

    # 非tensor参数数量不匹配
    NON_TENSOR_INPUTS_COUNT_MISMATCH = (
        _AUTO_GENERATED_CONVERTER_HEADER +
        "the number of torch non-tensor inputs greater than the AscendIR {ge_name} attrs. torch non-tensor inputs: {non_tensor_inputs}, AscendIR attrs: {ge_attrs}. " +
        _MANUAL_IMPLEMENTATION_SOLUTION
    )

    # 输出参数数量不匹配
    OUTPUTS_COUNT_MISMATCH = (
        _AUTO_GENERATED_CONVERTER_HEADER +
        "the number of torch outputs does not match the AscendIR {ge_name} outputs. torch outputs count: {outputs_count}, AscendIR {ge_name} outputs: {ge_outputs_count}. " +
        _MANUAL_IMPLEMENTATION_SOLUTION
    )

    # inplace输出参数数量不匹配
    INPLACE_COUNT_MISMATCH = (
        _AUTO_GENERATED_CONVERTER_HEADER +
        "the number of inplace inputs for torch does not match the AscendIR {ge_name} inplace. torch inplace count: {inplace_count}, AscendIR inplace count: {ge_inplace_count}. " +
        _MANUAL_IMPLEMENTATION_SOLUTION
    )

    # 参数数量不匹配
    ARGS_COUNT_MISMATCH = (
        _ARGS_CONVERSION_HEADER +
        "The AscendIR {op_type} expected {expected} args but got {actual}. " +
        _PARAMS_DEFINITION_CHECK_HINT
    )

    # 输入类型错误 - 必需参数
    INPUT_TYPE_REQUIRED = (
        _ARGS_CONVERSION_HEADER +
        "The AscendIR {op_type} input name: {name}, param type: {param_type}, type must be Tensor. " +
        _PARAMS_DEFINITION_CHECK_HINT
    )

    # 输入类型错误 - 动态参数
    INPUT_TYPE_DYNAMIC = (
        _ARGS_CONVERSION_HEADER +
        "The AscendIR {op_type} input name: {name}, param type: {param_type}, type must be list or tuple. " +
        _PARAMS_DEFINITION_CHECK_HINT
    )

    # 输入类型错误 - 可选参数
    INPUT_TYPE_OPTIONAL = (
        _ARGS_CONVERSION_HEADER +
        "The AscendIR {op_type} input name: {name}, param type: {param_type}, type must be Tensor or None. " +
        _PARAMS_DEFINITION_CHECK_HINT
    )

    # 输入类型定义非法
    INPUT_TYPE_ILLEGAL = (
        _ARGS_CONVERSION_HEADER +
        "The AscendIR {op_type} input name: {name}, param type: {param_type} is not legal, param type must be in: [required, dynamic, optional]. "
        "Please check the params definition of AscendIR {op_type}."
    )

    # 属性类型非法
    ATTR_TYPE_ILLEGAL = (
        _ARGS_CONVERSION_HEADER +
        "The AscendIR {op_type} attr type: {attr_type}, name: {name} is not legal, attr type must be in: {all_attr_types}. "
        "Please check the params definition of AscendIR {op_type}."
    )

    # 不支持torch type to ge type
    TORCH_TYPE_TO_GE_TYPE_UN_SUPPORT = (
        "Unsupported convert torch type {dtype} to ge type, "
        "please refer to the published Ascend Extension for PyTorch version documentation for guidance on usage, or submit an issue through the torchair community repository."
    )
