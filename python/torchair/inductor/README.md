## 安装
```SHELL
cd torchair/python/torchair/inductor
pip3 install -e .
```

## 测试
> 测试前需要在torchair/python/torchair/inductor/npu_extension_for_inductor/fuser/stub.py中对auto_fuse进行打桩，可以找qiduan要

> 注意stub中的芯片类型要和实际环境匹配
```shell
cd torchair/python/torchair/inductor/test
python3 test.py
```