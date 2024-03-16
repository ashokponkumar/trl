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

from tokenizers import Tokenizer
from typing import List

def detokenize(element, tokenizer: Tokenizer, column_names: List[str], **kwargs):    
    detokenized_data = {}
    for col in column_names:
        valid_token_ids = [ele for ele in element[col] if ele != -100]
        detokenized_data[col] = tokenizer.decode(valid_token_ids)
    
    return detokenized_data