""" Utility classes and functions related to MoCL (NAACL 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
sys.path.append(".")
sys.path.append("../")

from transformers import AutoConfig, T5Tokenizer, set_seed
from arguments import get_args

from utils.set_logger import set_logger
from model.modeling_t5 import T5ForConditionalGeneration
# from model.t5_continual import T5PrefixContinualForConditionalGeneration
from model.t5_continual_task_feature_with_contrastive_learning import T5PrefixContinualForConditionalGeneration
from training.trainer_continual_mtl_t5_feature import ContinualTrainerMTL
from tasks.mtl5.dataloader_mtl_t5_temp import DataLoaderMTL
import pdb

if __name__ == "__main__":
    args = get_args()
    model_args, data_args, training_args = args
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    logfile = os.path.join(training_args.output_dir, "log.txt")
    logger = set_logger(logfile)
        
    config_path = os.path.join(training_args.output_dir, f"configs.json")
    with open(config_path, "a", newline='\n') as f:
        f.write(f"\nmodel_args:\n {model_args}\n")
        f.write(f"\ndata_args:\n {data_args}\n")
        f.write(f"\ntraining_args:\n {training_args}\n")

    
    task_list = model_args.mtl_task_list.split('_')
    model_args.cl_language_list = model_args.mtl_task_list # TODO: make them consistent and delete this
    mode_list = ["train", "test"]
    task2labels = {
        'agnews': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'yelp': ['1', '2', '3', '4', '5'],
        'amazon': ['1', '2', '3', '4', '5'],
        'yahoo': ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music', 'Family & Relationships', 'Politics & Government'],
        'dbpedia': ['Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder', 'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'WrittenWork']
        }
    label_list = [[] for _ in task_list]
    num_labels = 0
    # pdb.set_trace()
    for ti, task in enumerate(task_list):
        label_list[ti] = task2labels[task]
        num_labels += len(label_list[ti])
    
    task2target_len = {
        'agnews': 2,
        'yelp': 2,
        'amazon': 2,
        'yahoo': 5,
        'dbpedia': 5
        }
    
    # Set seed before initializing model.
    seed = training_args.seed
    set_seed(seed)
    
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        revision=model_args.model_revision,
    )
    
    # print(config.classifier_match_embed, flush=True)
    
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    
    model = T5PrefixContinualForConditionalGeneration(
        training_args, 
        model_args, 
        model, 
        tokenizer,
        task_list,
        task2target_len,
    )
    
    # pdb.set_trace()
    
    dataloaders = DataLoaderMTL(
        data_args,
        training_args,
        task_list,
        label_list,
        mode_list,
        tokenizer,
        data_args.max_seq_length,
        data_args.pad_to_max_length,
        data_args.overwrite_cache,
        logger
    )
    
    train_dataloaders = {task: dataloaders[task]['train'] for task in task_list}
    dev_dataloaders = {task: dataloaders[task]['dev'] for task in task_list}
    test_dataloaders = {task: dataloaders[task]['test'] for task in task_list}
    
    
    # pdb.set_trace()
    
    trainer = ContinualTrainerMTL(
        training_args,
        model,
        logger,
        task_list,
        data_args.early_stopping_patience if data_args.early_stop else -1,
        tokenizer=tokenizer,
        learning_rate_list=data_args.learning_rate_list,
    )
    
    # print(trainer.feature_extractor(model, train_dataloaders),flush=True)
    
    trainer.train(
        train_dataloaders,
        dev_dataloaders,
        test_dataloaders,
    )