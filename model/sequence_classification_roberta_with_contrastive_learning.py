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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import model.task_feature_out as task_feature_out
import pickle
import pdb

from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from model.mtl_prefix_encoder import PrefixContinualMTLEncoder, PrefixQKVEncoder
from utils.utilities import mahalanobis


class RobertaPrefixContinualForSequenceClassification_contrastive_learning(RobertaPreTrainedModel):
    def __init__(self, config, task_limit):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob) 
        
        self.num_tasks = len(config.cl_language_list.split('_'))
        if config.task_specific_classifier:
            if type(config.label_list[0])==list:
                self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, len(labels))
                                                        for labels in config.label_list])
            else:
                self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, config.num_labels)
                                                    for _ in range(self.num_tasks)])
        else:
            self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        
        self.query_encoder = self.roberta
        # query and keys should have the same dimension
        config.key_dim = config.hidden_size
        self.query_size = config.hidden_size
        self.task_infer_acc = []
        
        self.embeds = {}
        self.feature_out = {}
        self.save_sample = [False] * self.num_tasks
        
        # task 별 feature vector를 계산하기 위해 embeds 값을 저장해둠
        for i in range(self.num_tasks):
            self.embeds[str(i)] = []
            self.feature_out[str(i)] = []
        
        print(self.config.compose_prompts)
        if self.config.compose_prompts:
            self.prefix_encoder = PrefixQKVEncoder(config, task_limit)
            self.embed_prototypes = [[] for _ in range(self.num_tasks)]
            self.till_embed_prototypes = [None for _ in range(self.num_tasks)]

        else:
            self.prefix_encoder = PrefixContinualMTLEncoder(config)
            
        if self.config.task_identify_epi:
            self.task_embeds = [[] for _ in range(self.num_tasks)]
            self.task_labels = [[] for _ in range(self.num_tasks)]
            
            self.task_means_over_classes = nn.ParameterList()
            # share covariance acrosss all tasks
            self.accumulate_shared_covs = nn.Parameter(torch.zeros(
                self.query_size, self.query_size), requires_grad=False)
            self.cov_inv = nn.Parameter(torch.ones(
                self.query_size, self.query_size), requires_grad=False)
 
        self.contrastive_learned = [False] * self.num_tasks
        
        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
                
        print('all param is {}M'.format(all_param/(1024*1024)))
        print('trainable param is {}M'.format(total_param/(1024*1024)))
    
    
    def get_prompt(self, batch_size, x_query=None, task_id=None, train=False, final=False, prompt_select_mode='local_compose', task_id_list=None, id_pred_on=False):
        input0 = x_query if self.config.compose_prompts else batch_size
        # print("ROBERTA model의 embed_prototypes: ", self.embed_prototypes, flush=True)
        outputs = self.prefix_encoder(input0, task_id, self.embed_prototypes, train, final, prompt_select_mode, task_id_list, id_pred_on)

        return outputs if self.config.compose_prompts else (outputs, 0.) # [ComposePrompts] / [vanilla CL & ProgressivePrompt & EPI without composition]

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_id=None,
        train=False,
        final=False, # final evaluation 
        prompt_select_mode='local_compose',
        classifier_select_mode='task_id',
        id_pred_on=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        x_query = None
        task_id_list=None
        if self.config.compose_prompts or self.config.task_identify_epi:
            x_embed = self.query_encoder(input_ids, attention_mask=attention_mask)[0]
            x_query = self.process_query(x_embed, attention_mask, train, task_id, labels)

            if self.save_sample[task_id] == False:
                with open(f"{self.config.output_dir}/{task_id}_x_query.pkl","wb") as f:
                    pickle.dump(x_query, f)
                self.save_sample[task_id] = True
                
            if final:
                if id_pred_on or classifier_select_mode != 'task_id': # pred task ids for prompt selection and/or classifier selection
                    if not (self.config.task_identify_epi or self.config.classifier_match_embed or self.config.classifier_match_key):
                        self.config.classifier_match_key = True
                    task_id_list, task_id_prob = self._get_pred_ids(x_query, task_id, final, train)
            
        past_key_values, match_loss = self.get_prompt(batch_size, x_query, task_id, train=train, final=final, prompt_select_mode=prompt_select_mode, task_id_list=task_id_list, id_pred_on=id_pred_on)

        prefix_attention_mask = torch.ones(batch_size, past_key_values[0].shape[-2]).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        if self.config.task_specific_classifier:
            self.num_labels = self.classifiers[task_id].out_features
            if train:
                logits = self.classifiers[task_id](pooled_output)
                
            else:
                if classifier_select_mode == 'task_id':
                    logits = self.classifiers[task_id](pooled_output)
                
                else:
                    self.num_labels = self.config.num_labels
                    logits = (torch.ones(batch_size, self.config.num_labels).to(x_query.device)) *(-float('inf'))
                    labels = self._add_label_offsets(labels, task_id)
                    
                    for t_id in range(self.num_tasks):
                        start, end = self.task_range[t_id]
                        if classifier_select_mode == 'compose':
                            logits[:, start:end] = self.classifiers[t_id](pooled_output) * task_id_prob[:, t_id][:, None]
                        else: # classifier_select_mode = 'direct':
                            logits[:, start:end] = self.classifiers[t_id](pooled_output)
                    
                    if classifier_select_mode == 'top1':
                        for b_id, t_id in enumerate(task_id_list): # [bs, num_tasks]
                            start, end = self.task_range[t_id]
                            logits[b_id][:start] = -float('inf')
                            logits[b_id][end:] = -float('inf')
                    
        else:
            logits = self.classifier(pooled_output)
            self.num_labels = self.classifier.out_features
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if train:
            try:
                self.prefix_encoder.writer.add_scalar(f'task_loss/task_{task_id}', loss.cpu(), self.prefix_encoder.steps[task_id])
                self.prefix_encoder.writer.add_scalar(f'total_loss/task_{task_id}', (loss+match_loss).cpu(), self.prefix_encoder.steps[task_id])
            except:
                pass
            
        return SequenceClassifierOutput(
            loss=loss+match_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), labels
        
    
    # def process_query(self, embed, attention_mask, train, task_id, labels=None):
    #     if 'cls_token_embed' in self.config.query_encoder_type:
    #         x_query = embed[:, 0, :]
    #     elif self.config.query_encoder_type == 'avg_all_embed':
    #         x_query = embed.mean(dim=1)
    #     elif self.config.query_encoder_type == 'avg_word_embed':
    #         x_query = torch.sum(embed * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        
    #     # store embedding prototypes if required
        
    #     if train and self.config.classifier_match_embed and type(self.embed_prototypes[task_id]) == list:
    #         self.embed_prototypes[task_id].extend(x_query.tolist())

    #     if train and self.config.task_identify_epi and len(self.task_means_over_classes)<task_id+1:
    #         # only add once is enough
    #         self.task_embeds[task_id].extend(x_query.tolist())
    #         self.task_labels[task_id].extend(labels.tolist())
            
    #     return x_query
    
    # def process_query(self, embed, attention_mask, train, task_id, labels=None):
    #     # Using 'avg_all_embed' by default
    #     x_query = embed.mean(dim=1)

    #     # embed_prototypes를 통해 각 task의 prototype을 저장함, 해당 prototype과 x_query 간의 유사도를 통해 초기화에 사용.
    #     # embed_prototypes는 get_task_embeds()를 통해 구함, task가 추가되면 contrastive learning을 통해 기존의 task들의 prototype과 멀어지게 학습한 후 결과 뱉음
    #     # task 별 100개의 embed 값이 모이면 100개의 batch들을 통해 각 task의 prototype을 구함
                
    #     while len(self.embeds[str(task_id)]) < 100:
    #         self.embeds[str(task_id)].append(embed)
    #         print("embeds 값들 저장 중", flush=True)
        
    #     if train and self.config.classifier_match_embed:

    #         if self.embed_prototypes[task_id] is not None:
                
    #             if (len(self.embeds[str(task_id)]) == 100) & (self.contrastive_learned[task_id] == False):
    #                 print("embed_prototypes contrastive learning으로 udpate", flush=True)
    #                 task_feature_out.get_task_feature_vector(self.embeds, task_id, self.config.output_dir)
    #                 # feature_out = pickle.load(open(f"til_{task_id}_task_feature_vector.pkl", "rb"))
    #                 feature_out = pickle.load(open(f"{self.config.output_dir}/til_{task_id}_task_feature_vector_order_2.pkl", "rb"))

    #                 # pdb.set_trace()
    #                 for per_task in range(0, task_id+1):
    #                     self.embed_prototypes[per_task] = (sum(feature_out[f"{per_task}"])/len(feature_out[f"{per_task}"])).mean(dim=0)
                        
    #                 print("embed_prototypes contrastive learning으로 update 완료", flush=True)
    #                 self.contrastive_learned[task_id] = True
                    
    #             elif len(self.embeds[str(task_id)]) > 100:
    #                 self.embed_prototypes[task_id] = self.embed_prototypes[task_id]
                                
    #         else:
    #             # self.embed_prototypes[task_id] = self.get_task_embeds(embed)
    #             self.embed_prototypes[task_id] = x_query.mean(dim=0)
    #             print("embed_shape: ",embed.shape)
    #             print("task_id: ", task_id)
    #             print("embed_prototypes_shape: ", self.embed_prototypes[task_id].shape)
                
    #             # for i in range(self.task_list):
    #             #     task_feature_out.get_task_feature_vector(embed, labels)
                
    #             # self.embed_prototypes[task_id] = task_feature_out.get_task_feature_vector(embed, task_id, labels)
    #             # print(f"task {task_id} prototype: {self.embed_prototypes[task_id]}")
    #             # self.embed_prototypes[task_id] = x_query.mean(dim=0)
            
    #     if train and self.config.task_identify_epi and self.task_embeds[task_id]==[]:
    #         # only add once is enough
    #         # self.task_embeds[task_id].extend(x_query.tolist())
    #         # self.task_embeds[task_id].extend(result_of_cl)
    #         self.task_labels[task_id].extend(labels.tolist())
            
    #     return x_query
    
    def process_query(self, embed, attention_mask, train, task_id, labels=None):
        # Using 'avg_all_embed' by default
        x_query = embed.mean(dim=1)
        # result_of_cl = task_feature_out.get_task_feature_vector(embed, task_id, labels)
        
        # print("##########")
        # print(train and self.config.classifier_match_embed)
        
        # store embedding prototypes if required
        ############
        # embed_prototypes를 통해 각 task의 prototype을 저장함, 해당 prototype과 x_query 간의 유사도를 통해 초기화에 사용.
        # embed_prototypes는 get_task_embeds()를 통해 구함, task가 추가되면 contrastive learning을 통해 기존의 task들의 prototype과 멀어지게 학습한 후 결과 뱉음
        # task 별 100개의 embed 값이 모이면 100개의 batch들을 통해 각 task의 prototype을 구함
        
        
        # 30개의 embeds가 모이면 contrastive learning을 통해 prototype을 업데이트함
        # while len(self.embeds[str(task_id)]) < sample_embeds_number:
        #     self.embeds[str(task_id)].append(embed)
        
        if self.config.classifier_match_embed:
            # if self.embed_prototypes[task_id] is not None:
            #     print("embed_prototypes 업데이트 중", flush=True)
            #     steps = self.prefix_encoder.steps[task_id]
            #     # get_task_embeds()는 task가 추가되면 
            #     output_dir = "first_embeds"
            #     os.makedirs(output_dir, exist_ok=True)
            #     pickle.dump(self.embed_prototypes[task_id], open(f"first_embeds/{task_id}.pkl", "wb"))
            
            if task_id!=0 and self.contrastive_learned[task_id] == False:
                
                # print("embed_prototypes lengh: ", len(self.embed_prototypes[int(task_id)]))
                # pdb.set_trace()
                # print("embeds 값의 길이",len(self.embeds[str(task_id)]))
                # print(self.contrastive_learned[task])
                
                # if task_id!=0 & (len(self.embeds[str(task_id)]) == 30) & (self.contrastive_learned[task_id] == False):
                #     print("embed_prototypes contrastive learning으로 udpate", flush=True)
                #     # task_feature_out.get_task_feature_vector(self.embeds, task_id, self.config.output_dir, epochs=10)
                #     # feature_out = pickle.load(open(f"til_{task_id}_task_feature_vector.pkl", "rb"))
                feature_out = pickle.load(open(f"{self.config.output_dir}/til_{task_id}_task_feature_vector.pkl", "rb"))
                # pdb.set_trace()
                
                for per_task in range(0, task_id+1):
                    self.embed_prototypes[per_task] = torch.cat(feature_out[str(per_task)]).mean(dim=0)
                
                self.till_embed_prototypes[task_id] = self.embed_prototypes[task_id]
                        
                self.contrastive_learned[task_id] = True
                              
            else:
                self.embed_prototypes[task_id] = x_query.mean(dim=0)
                self.till_embed_prototypes[task_id] = self.embed_prototypes[task_id]
                
            file_path = f"{self.config.output_dir}/til_{task_id}_embed_prototypes.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(self.embed_prototypes, f)
            
        if self.config.task_identify_epi and self.task_embeds[task_id]==[]:
            # only add once is enough
            # self.task_embeds[task_id].extend(x_query.tolist())
            # self.task_embeds[task_id].extend(result_of_cl)
            self.task_labels[task_id].extend(labels.tolist())
            
        return x_query
    
    def _get_pred_ids(self, x_query, task_id, final, train, log_id='cil'):
        if self.config.classifier_match_embed:
            if train:
                embed_prototypes = torch.stack(self.embed_prototypes)
                cos_sim = torch.cosine_similarity(x_query.unsqueeze(1), embed_prototypes.unsqueeze(0), dim=-1) # [bs, num_tasks]
                prob = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) # if not self.config.direct_compose else cos_sim; comment this because for the prediction, softmax is required and scale is necessary in this case.
                pred_ids = torch.argmax(prob, dim=-1) # [bs]
            
            else:
                till_embed_prototypes = torch.stack(self.till_embed_prototypes[:task_id+1])
                
                cos_sim = torch.cosine_similarity(x_query.unsqueeze(1), till_embed_prototypes.unsqueeze(0), dim=-1) # [bs, num_tasks]
                prob = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) # if not self.config.direct_compose else cos_sim; comment this because for the prediction, softmax is required and scale is necessary in this case.
                pred_ids = torch.argmax(prob, dim=-1) # [bs]
            
        elif self.config.classifier_match_key:
            K = self.prefix_encoder.keys
            cos_sim = torch.cosine_similarity(x_query.unsqueeze(1), K.unsqueeze(0), dim=-1) # [bs, num_tasks]
            prob = nn.functional.softmax(cos_sim*self.config.softmax_match_scale, dim=-1) # if not self.config.direct_compose else cos_sim
            pred_ids = torch.argmax(prob, dim=-1) # [bs]
            
        elif self.config.task_identify_epi: # epi has the lowest priority
            pred_ids, prob = self._get_epi_ids(x_query)
            log_id='epi'
                    
        tg_ids = torch.ones_like(pred_ids).to(pred_ids.device) * task_id
        accuracy = (pred_ids == tg_ids).float()
        step = self.prefix_encoder.steps_final[task_id]
        self.prefix_encoder.writer.add_scalar(f'key_match_acc/{log_id}_task_{task_id}', accuracy.mean(), step)
        self.task_infer_acc.append(accuracy.cpu().mean().item())
        return pred_ids, prob
    
    
    def _get_epi_ids(self, query):
        scores_over_tasks = []
        for mean_over_classes in self.task_means_over_classes:
            num_labels = mean_over_classes.shape[0]
            score_over_classes = []
            for l in range(num_labels):
                score = mahalanobis(query, mean_over_classes[l], self.cov_inv, norm=2)
                score_over_classes.append(score)
            
            # [num_labels, bs]
            score_over_classes = torch.stack(score_over_classes)
            score, _ = score_over_classes.min(dim=0)
            
            scores_over_tasks.append(score)
        
        # [num_tasks, bs]
        scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
        _, ids = torch.min(scores_over_tasks, dim=0)
        
        prob_over_tasks = (scores_over_tasks*(-1)).T.softmax(dim=-1)
        
        return ids, prob_over_tasks
    
    
    def _add_label_offsets(self, labels, task_id):
        offset = 0
        for i in range(task_id):
            offset += len(self.config.label_list[i])
        return labels + offset