#!/bin/bash
# coding=utf-8

diffusers_path=$(pip3 show diffusers | grep Location | awk '{print $2}')
cp diffusers_npu.patch ${diffusers_path}/diffusers
cd ${diffusers_path}/diffusers

patch -p4 < diffusers_npu.patch