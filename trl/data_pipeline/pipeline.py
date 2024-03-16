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

import yaml
from typing import Union, Tuple

from datasets import Dataset

def data_pipeline(self, data_config: Union[str, dict]) -> Tuple[Dataset, Dataset, Dataset] :
    if isinstance(data_config, str):
        with open(data_config) as f:
            data_config = yaml.load(f)

    dataset_with_splits = None
    self.train_datasets = None
    self.eval_datasets = None
    self.test_datasets = None
    
    with PartialState().local_main_process_first():
        for dataset_config in data_config["datasets"]:
            logger.info("Loading %s" % (dataset_config["name"]))
            kwargs = {}
            data_type = None
            if "loader_arguments" in dataset_config:
                kwargs = dataset_config["loader_arguments"]
                if "data_files" in kwargs:
                    if isinstance(kwargs["data_files"], Dict):
                        for k, v in kwargs["data_files"].items():
                            kwargs["data_files"][k] = fix_data_paths(v)
                    else:
                        kwargs["data_files"]=fix_data_paths(kwargs["data_files"])
                if "data_dir" in kwargs:
                    kwargs["data_dir"]=fix_data_paths(kwargs["data_dir"])
                if ("split" not in kwargs) and isinstance(kwargs["data_files"], str):
                    kwargs["split"] = "train"
                if "data_files" in kwargs and "data_type" not in kwargs:
                    data_type = get_loader_for_filepath(kwargs["data_files"])
                else:
                    data_type = kwargs["data_type"]
                    del kwargs["data_type"]
                # if data_type != "text" and "keep_linebreaks" in kwargs:
                #     del kwargs["keep_linebreaks"]
            raw_dataset = datasets.load_dataset(data_type, **kwargs)
            
            if isinstance(raw_dataset, datasets.IterableDataset):
                raw_datasets = datasets.IterableDatasetDict()
                if self.eval_datasets == None:
                    self.eval_datasets = datasets.IterableDatasetDict()
                if self.test_datasets == None:
                    self.test_datasets = datasets.IterableDatasetDict()
            else:
                raw_datasets = datasets.DatasetDict()
                if self.eval_datasets == None:
                    self.eval_datasets = datasets.DatasetDict()
                if self.test_datasets == None:
                    self.test_datasets = datasets.DatasetDict()
            splitName = "train"
            if "split" in kwargs:
                splitName =  kwargs["split"]
            elif (isinstance(kwargs["data_files"], dict) and len(kwargs["data_files"]) > 0):
                splitName = list(kwargs["data_files"].keys())[0]  
            
            if isinstance(kwargs["data_files"], dict) or isinstance(kwargs["data_files"], List):
                raw_datasets = raw_dataset
            else:
                raw_datasets[splitName] = raw_dataset
            # if self.fm_args.data_debug:
            #     logger.info(f"Sample data : {raw_datasets[splitName][0]}")
            kwargs = {}
            if "splitter_arguments" in dataset_config:
                kwargs = dataset_config["splitter_arguments"]
                if "seed" not in kwargs or kwargs["seed"] is None:
                    kwargs["seed"] = self.training_args.seed
                raw_datasets = raw_dataset.train_test_split(**kwargs)
                raw_datasets["eval"] = raw_datasets["test"]
                raw_datasets.pop("test")
            if "data_handlers" in dataset_config:
                if self.fm_args.data_debug:
                    data_handler_debug = {
                            "name": "detokenize",
                            "arguments" : {
                                            "remove_columns" : "all",
                                            "batched" : False
                                        }
                            }
                    dataset_config["data_handlers"].append(data_handler_debug)
                for data_handler in dataset_config["data_handlers"]:
                    kwargs = {}
                    if "arguments" in data_handler:
                        kwargs = data_handler["arguments"]
                    if "batched" not in kwargs:
                        kwargs["batched"]=True

                    column_names = raw_datasets[splitName].column_names
                    # remove __content__ from all processing
                    if "__content__" in column_names:
                        column_names.remove("__content__")
                    if "remove_columns" not in kwargs:
                        kwargs["remove_columns"] = None
                    if kwargs["remove_columns"]=="all":
                        kwargs["remove_columns"] = column_names

                    if "fn_kwargs" not in kwargs:
                        kwargs["fn_kwargs"] = {}
                    if isinstance(raw_datasets, datasets.DatasetDict) and "num_proc" not in kwargs:
                        kwargs["num_proc"]=os.cpu_count()
                    kwargs["fn_kwargs"]["tokenizer"] = self.tokenizer
                    kwargs["fn_kwargs"]["block_size"] = self.fm_args.block_size
                    kwargs["fn_kwargs"]["column_names"] = column_names
                    kwargs["fn_kwargs"]["model_path"] = self.fm_args.base_model_path
                    logger.info("Loaded raw dataset : {raw_datasets}")
                    raw_datasets=raw_datasets.map(self.data_handlers[data_handler["name"]], **kwargs)
                    if self.fm_args.data_debug:
                        data_handler_name = data_handler["name"]
                        logger.info(f"Sample data after {data_handler_name}: {raw_datasets[splitName][0]}")
                    if data_handler["name"] == "preprocess_prompt":
                        self.task = kwargs["fn_kwargs"]["task"]
            
            if "eval" in raw_datasets:
                self.eval_datasets["eval_" + dataset_config["name"]] = raw_datasets["eval"]
            if "test" in raw_datasets:
                self.test_datasets["test_" + dataset_config["name"]] = raw_datasets["test"]
            
            if dataset_with_splits is None:
                dataset_with_splits = raw_datasets
            else:
                for k in raw_datasets.keys():
                    if k in dataset_with_splits:
                        dataset_with_splits[k] = datasets.concatenate_datasets([dataset_with_splits[k], raw_datasets[k]])
                    else:
                        dataset_with_splits[k] = raw_datasets[k]            
    
    if "train" in dataset_with_splits:
        self.train_datasets = dataset_with_splits["train"]
    
    if "eval" in dataset_with_splits:
        if len(self.eval_datasets) > 1:
            self.eval_datasets["eval"] = dataset_with_splits["eval"]
        else:
            self.eval_datasets = dataset_with_splits["eval"]
    
    if "test" in dataset_with_splits:
        if len(self.test_datasets) > 1:
            self.test_datasets["test"] = dataset_with_splits["test"]
        else:
            self.test_datasets = dataset_with_splits["test"]
    
    if len(self.eval_datasets) == 0:
        self.eval_datasets = None 
    if len(self.test_datasets) == 0:
        self.test_datasets = None
    