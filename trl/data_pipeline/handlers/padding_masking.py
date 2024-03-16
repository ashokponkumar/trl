# Copyright 2024 The HuggingFace Team. All rights reserved.
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

def padding_masking(element, block_size, padding_side, tokenizer, **kwargs):
    inputs = element["inputs"]
    outputs = element["outputs"]
    
    if padding_side == 'left':
        input_ids = [tokenizer.pad_token_id] * (block_size - len(inputs)) + inputs
        attention_mask = [0] * (block_size - len(inputs)) + [1] * len(inputs)
        labels = [-100] * (block_size - len(outputs)) + outputs
    else:
        input_ids = inputs + [tokenizer.pad_token_id] * (block_size - len(inputs))
        attention_mask = [1] * len(inputs) + [0] * (block_size - len(inputs))
        labels = [-100] * (len(inputs) - len(outputs)) + outputs + [-100] * (block_size - len(inputs))
            
    return {
        "input_ids" : input_ids,
        # "attention_mask" : attention_mask,
        "labels" : labels
    }