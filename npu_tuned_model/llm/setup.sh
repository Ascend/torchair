#!/bin/bash
# coding=utf-8

pip3 install transformers==4.31.0
transformers_path=$(pip3 show transformers|grep Location|awk '{print $2}')
cp models/common/utils.py ${transformers_path}/transformers/generation/