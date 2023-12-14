from dataclasses import dataclass
from typing import List
import json

Code = str


class OpProto:
    @dataclass
    class Desc:
        name: str
        param_type: str
        format: List[str]
        type: List[str]

    def __init__(self, json_str: str):
        self.json = json.loads(json_str)
        self.name = self.json['op']
        self.inputs = [OpProto.Desc(**desc) for desc in self.json["input_desc"]]
        self.outputs = [OpProto.Desc(**desc) for desc in self.json["output_desc"]]

    def __str__(self):
        return str(self.json)


@dataclass
class OpCode:
    proto: OpProto
    tiling: Code
    host: Code
    device: Code
