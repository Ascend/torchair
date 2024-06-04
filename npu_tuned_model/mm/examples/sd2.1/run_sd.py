# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import logging
import time
import random
import torch
import numpy as np
import torch.distributed
import torch_npu
import diffusers
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="sd run parameters")
    parser.add_argument('--model_path', type=str, help="Location of model weights")
    parser.add_argument('--dynamo', action='store_true', default=False, help="Whether use torch dynamo mode")
    parser.add_argument('--soc_version', type=str, default="310P", help="Ascend SOC Version")
    parser.add_argument('--local_rank', type=int, default=0, help="Ascend Local Rank")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--FA', action='store_true', default=False, help="Whether use npu flash attention")
    parser.add_argument('--TOME', action='store_true', default=False, help="Whether use Token Merge")
    parser.add_argument('--DC', action='store_true', default=False, help="Whether use DC")
    parser_args = parser.parse_args()
    return parser_args


def init_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


def run_sd(args):
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    device = torch.device("npu:%s" % args.local_rank)
    init_seed(args.seed)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.distributed.init_process_group(backend="hccl", world_size=world_size, rank=args.local_rank)
    pipe.to("npu:%s" % args.local_rank)

    if args.FA:
        from models.npu.attention_processor import AttnProcessor2_0, AttnProcessor2_0Tome
        pipe.unet.set_attn_processor(AttnProcessor2_0())

    if args.TOME:
        if not args.dynamo:
            raise NotImplementedError("Token Merge is not supported in eager mode")
        pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())
        pipe.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())
        pipe.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())
        pipe.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())

    
    if args.dynamo:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        npu_backend = tng.get_npu_backend(compiler_config=config)

        '''
        use weight nz
        '''
        tng.experimental.inference.use_internal_format_weight(pipe.text_encoder)
        tng.experimental.inference.use_internal_format_weight(pipe.unet)
        tng.experimental.inference.use_internal_format_weight(pipe.vae.decoder)
        '''
        torch dynamo
        '''
        pipe.text_encoder = torch.compile(pipe.text_encoder, backend=npu_backend)
        pipe.unet = torch.compile(pipe.unet, backend=npu_backend)
        pipe.unet.forward_deepcache = torch.compile(pipe.unet.forward_deepcache, backend=npu_backend)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend=npu_backend)

    prompt = "a photo of an astronaut riding a horse on mars"

    if args.DC:
        '''
        warmup
        '''
        image = pipe(prompt, cache_interval=2, cache_layer_id=0, cache_block_id=0).images[0] 
        image = pipe(prompt, cache_interval=2, cache_layer_id=0, cache_block_id=0).images[0] 

        start_time = time.time()
        image = pipe(prompt, cache_interval=2, cache_layer_id=0, cache_block_id=0).images[0]
        inference_time = time.time() - start_time
    else:
        '''
        warmup
        '''
        image = pipe(prompt).images[0] 
        image = pipe(prompt).images[0] 

        start_time = time.time()
        image = pipe(prompt).images[0]
        inference_time = time.time() - start_time

    print("inference cost {}s".format(inference_time))

        
    image.save("astronaut_rides_horse.png")
    
if __name__ == "__main__":
    configs = parse_args()
    run_sd(configs)
    logging.info("model run success")
