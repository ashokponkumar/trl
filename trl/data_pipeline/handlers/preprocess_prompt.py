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

from backend import utilities as u

util = u.Utilities(None, None)
util.set_logger(util.get_logger())

def preprocess_prompt(element, **kwargs):
    element["__content__"] = dict(element.copy())
    preprocessed_prompt = util.get_processed_prompt(kwargs["task"], kwargs["model_path"], element["input"]["prompt"])
    p = {
        "prompt": preprocessed_prompt,
        "task": kwargs["task"]
    }
    element["__content__"]["predictions"].append(p)
    element["prompt"] =  preprocessed_prompt
    return element