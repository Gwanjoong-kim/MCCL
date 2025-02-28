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
from training.trainer_continual_mtl_t5 import ContinualTrainerMTL
from tasks.mtl5.dataloader_mtl_t5 import DataLoaderMTL
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
    
    
    train_loader, train_batch = _prepare_dataloaders(train_dataloaders)
    
    query_encoder = model.encoder
    
    x_embed = query_encoder(
        inputs_embeds = inputs_embeds,
        attention_mask = attention_mask,
        head_mask=None,  
        output_attentions=None,  
        output_hidden_states=None, 
        return_dict=None,  
    ).last_hidden_state
    
    model.process_query(embed, task_id)