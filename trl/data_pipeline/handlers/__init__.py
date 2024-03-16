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

from .tokenize import tokenize
from .group_texts import group_texts
from .padding_masking import padding_masking
from .detokenize import detokenize
from .preprocess_prompt import preprocess_prompt
from .prepare_predict_input import prepare_predict_input
from .preprocess_raw_prompt import preprocess_raw_prompt

default_data_handlers = { 
    "tokenize": tokenize, 
    "group_texts" : group_texts, 
    "padding_masking": padding_masking,
    "detokenize": detokenize,
    "preprocess_prompt": preprocess_prompt,
    "prepare_predict_input": prepare_predict_input,
    "preprocess_raw_prompt": preprocess_raw_prompt
    }
