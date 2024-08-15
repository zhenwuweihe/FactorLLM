import os
import torch
from typing import Dict
import lightning as L
import glob

import torch.utils
from configuration_llama import LlamaConfig
# from new_modeling_llama import LlamaForCausalLM_stu, LlamaForCausalLM
from new_model_KD import LlamaForCausalLM_stu, LlamaForCausalLM
# from transformers import LlamaForCausalLM
from packed_dataset import PackedDataset
import transformers
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LambdaLR

import sklearn

from sklearn import preprocessing

import numpy as np
# from k_means_constrained import KMeansConstrained

route_weight = 0.05
balance_weight = 0.01
learning_rate = 4e-5  # [1e-6, 7e-4, 3e-4, 2e-5, 4e-5, 5e-5,6e-5, 8e-5]
weight_decay = 1e-5
beta1 = 0.9
beta2 = 0.95  
grad_clip = 1.0
decay_lr = True
# min_lr = 1e-6
warmup_step = 500
total_step = 10000 # 10000
min_lr_ratio = 0.001
# save_model_list = [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000]
# save_model_list = [0, 1000, 2000, 4000, 8000, 10000, 15000, 20000, 25000, 30000, 50000]
# os.environ["RAIK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["http_proxy"] = "http://127.0.0.1:15777"
# os.environ["https_proxy"] = "http://127.0.0.1:15777"
def warmup_cosine_annealing_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = torch.tensor(float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(torch.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

def truncate_batch(batch, seq_length):
    for k, v in batch.items():  # k 是 key , v 是 value
        if v.size(1) > seq_length:
            batch[k] = v[:, :seq_length]  # 把里面所有长度长的全部裁剪
    return batch
import copy
class LightningLlamaModel(L.LightningModule):
    def __init__(self, model, weight_origin, arg, max_seq_len=2048):
        super().__init__()
        self.model = model
        # print(self.model)
        # raise "stop"
        self.max_seq_len = max_seq_len
        self.arg = arg
        print(f"arg = {arg}")
        # teacher_model = get_teacher_model(arg)
        # print(teacher_model)
        # raise "stop"
        # print(f"model == {model}")
        # print(f"teacher_model == {teacher_model}")


        self.model.model.embed_tokens.weight.data = weight_origin["model.embed_tokens.weight"]
        self.model.model.norm.weight.data = weight_origin["model.norm.weight"]
        self.model.lm_head.weight.data = weight_origin["lm_head.weight"]


        self.model.model.embed_tokens.weight.requires_grad = False
        self.model.model.norm.weight.requires_grad = False
        self.model.lm_head.weight.requires_grad = False

        # for i in range(22):
        #     print(i, self.model.model.layers[i].self_attn.q_proj.weight.dtype)
        #     print(i, teacher_model.model.layers[i].self_attn.q_proj.weight.dtype)
        # for key, value in weight_origin.items():
        #     print(key, value.shape)
        # raise "stop"
        # print(self.model)
        # print(self.model.model.layers[0].mlp[0])
        # print(self.model.model.layers[0].mlp[0].gate_proj.weight.shape)
        # print(self.model.model.layers[0].mlp[0].act_fn.weight)
        # raise "stop"
        for i in range(22):
            self.model.model.layers[i].self_attn.q_proj.weight.data = weight_origin[f"model.layers.{i}.self_attn.q_proj.weight"]
            self.model.model.layers[i].self_attn.k_proj.weight.data = weight_origin[f"model.layers.{i}.self_attn.k_proj.weight"]
            self.model.model.layers[i].self_attn.v_proj.weight.data = weight_origin[f"model.layers.{i}.self_attn.v_proj.weight"]
            self.model.model.layers[i].self_attn.o_proj.weight.data = weight_origin[f"model.layers.{i}.self_attn.o_proj.weight"]
            self.model.model.layers[i].mlp.gate_proj.weight.data = weight_origin[f"model.layers.{i}.mlp.gate_proj.weight"]
            self.model.model.layers[i].mlp.up_proj.weight.data = weight_origin[f"model.layers.{i}.mlp.up_proj.weight"]
            self.model.model.layers[i].mlp.down_proj.weight.data = weight_origin[f"model.layers.{i}.mlp.down_proj.weight"]
            self.model.model.layers[i].post_attention_layernorm.weight.data = weight_origin[f"model.layers.{i}.post_attention_layernorm.weight"]
            self.model.model.layers[i].input_layernorm.weight.data = weight_origin[f"model.layers.{i}.input_layernorm.weight"]
            
            self.model.model.layers[i].self_attn.q_proj.weight.requires_grad = False
            self.model.model.layers[i].self_attn.k_proj.weight.requires_grad = False
            self.model.model.layers[i].self_attn.v_proj.weight.requires_grad = False
            self.model.model.layers[i].self_attn.o_proj.weight.requires_grad = False
            self.model.model.layers[i].mlp.gate_proj.weight.requires_grad = False
            self.model.model.layers[i].mlp.up_proj.weight.requires_grad = False
            self.model.model.layers[i].mlp.down_proj.weight.requires_grad = False
            self.model.model.layers[i].post_attention_layernorm.weight.requires_grad = False
            self.model.model.layers[i].input_layernorm.weight.requires_grad = False





        

            # gate_weight = weight_origin[f"model.layers.{i}.mlp.gate_proj.weight"].numpy() # [5632, 2048]
            # up_weight = weight_origin[f"model.layers.{i}.mlp.up_proj.weight"].numpy() # [5632, 2048]
            # down_weight = weight_origin[f"model.layers.{i}.mlp.down_proj.weight"].transpose(0,1).numpy()  
            # gate_weight_norm = preprocessing.normalize(gate_weight)
            # up_weight_norm = preprocessing.normalize(up_weight)
            # down_weight_norm = preprocessing.normalize(down_weight)
            # gate_kmeans = KMeansConstrained(n_clusters=4, size_min=5632//4, size_max=5632//4, random_state=0).fit(gate_weight_norm, None).labels_
            # up_kmeans = KMeansConstrained(n_clusters=4, size_min=5632//4, size_max=5632//4, random_state=0).fit(up_weight_norm, None).labels_
            # down_kmeans = KMeansConstrained(n_clusters=4, size_min=5632//4, size_max=5632//4, random_state=0).fit(down_weight_norm, None).labels_

            # self.model.model.layers[i].mlp.route1.weight.requires_grad = False
            # self.model.model.layers[i].mlp.route1.bias.requires_grad = False
            # # k-mean聚类初始化
            # for j in range(4):
            #     # self.model.model.layers[i].mlp.gate_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.gate_proj.weight"][[k for k, x in enumerate(gate_kmeans) if x == j], :]
            #     # print("+++++++++++++++")
            #     # self.model.model.layers[i].mlp.up_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.up_proj.weight"][[k for k, x in enumerate(up_kmeans) if x == j], :]
            #     # print("-------------------------")
            #     # self.model.model.layers[i].mlp.down_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.down_proj.weight"][:, [k for k, x in enumerate(down_kmeans) if x == j]]
            #     # print("******************************")
            #     self.model.model.layers[i].mlp.gate_proj_list[j].weight.requires_grad = False
            #     self.model.model.layers[i].mlp.up_proj_list[j].weight.requires_grad = False
            #     self.model.model.layers[i].mlp.down_proj_list[j].weight.requires_grad = False

        
            # 将tinyllama每一层参数拆分
            # random.seed(2613)
            index = [i for i in range(5632)]
            random.shuffle(index)
            print(self.model.model.layers[0].mlp.expert_num)
            print(5632 // self.model.model.layers[0].mlp.expert_num)
            hidden_dim = 5632 // self.model.model.layers[0].mlp.expert_num
            # raise "stop"
            for j in range(self.model.model.layers[0].mlp.expert_num):
                # # print(teacher_model.model.layers[i].mlp.gate_proj.weight.data.shape)
                # # print(self.model.model.layers[i].mlp.down_proj_list[j].weight.data.shape)
                # # self.model.model.layers[i].mlp.gate_proj_list[j].weight.data =  teacher_model.model.layers[i].mlp.gate_proj.weight.data[j * hidden_dim: (j+1) * hidden_dim, :]
                # # self.model.model.layers[i].mlp.up_proj_list[j].weight.data =  teacher_model.model.layers[i].mlp.up_proj.weight.data[j * hidden_dim: (j+1) * hidden_dim, :]
                # # self.model.model.layers[i].mlp.down_proj_list[j].weight.data =  teacher_model.model.layers[i].mlp.down_proj.weight.data[:, j * hidden_dim: (j+1) * hidden_dim]
                self.model.model.layers[i].mlp.gate_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.gate_proj.weight"][j * hidden_dim: (j+1) * hidden_dim, :]
                self.model.model.layers[i].mlp.up_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.up_proj.weight"][j * hidden_dim: (j+1) * hidden_dim, :]
                self.model.model.layers[i].mlp.down_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.down_proj.weight"][:, j * hidden_dim: (j+1) * hidden_dim]
                # self.model.model.layers[i].mlp.gate_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.gate_proj.weight"][index[j * hidden_dim: (j+1) * hidden_dim], :]
                # self.model.model.layers[i].mlp.up_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.up_proj.weight"][index[j * hidden_dim: (j+1) * hidden_dim], :]
                # self.model.model.layers[i].mlp.down_proj_list[j].weight.data =  weight_origin[f"model.layers.{i}.mlp.down_proj.weight"][:, index[j * hidden_dim: (j+1) * hidden_dim]]
        # torch.save(self.model.state_dict(), "/vepfs-sha/dongmh/code/ckpt/test/pytorch_model.bin")
        # raise "stop"
        print("initialing model task has been finished")
        
        
        
        

    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        # batch 有{'input_ids', 'attention_mask', 'labels'}
        batch = truncate_batch(batch, self.max_seq_len)



        outputs, sum_route_loss, gt_hidden_states, sum_balance_loss = self(**batch)
        loss = outputs.loss
        self.clip_gradients(self.optimizers(), gradient_clip_val=grad_clip, gradient_clip_algorithm="norm")
        self.log("train_loss", loss, prog_bar=True)
        # self.log("sum_route_loss", sum_route_loss, prog_bar=True)
        # return loss + route_weight * sum_route_loss + sum_balance_loss * balance_weight
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = truncate_batch(batch, self.max_seq_len)


        outputs, sum_route_loss, gt_hidden_states, sum_balance_loss = self(**batch)
        loss =  route_weight * sum_route_loss + outputs.loss + sum_balance_loss * balance_weight
        # self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss", outputs.loss, prog_bar=True)
        # return loss
        return outputs.loss
    
    def on_validation_end(self) -> None:
        # if self.global_step > 49000:
        if True:
            # if self.global_step in save_model_list:
            weight = self.model.state_dict()
            new_weight = {}
            for key, value in weight.items():
                # new_key = key[6:]
                new_key = key
                new_weight[new_key] = value
            os.makedirs(f"/vepfs-sha/dongmh/metric_test/{self.arg.experiment_name}", exist_ok=True)
            torch.save(new_weight, f"/vepfs-sha/dongmh/metric_test/{self.arg.experiment_name}/pytorch_model_epoch{self.current_epoch}_step{self.global_step}.bin")

            
        

        return None
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False)
        scheduler = warmup_cosine_annealing_scheduler(
            optimizer, warmup_steps=warmup_step,
            total_steps=total_step, min_lr_ratio=min_lr_ratio
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
# import json
# from transformers import LLaMAForCausalLM, LLaMAConfig

# import llama_two_model
def get_teacher_model(args):
    print(f'loading base model {args.model_name_or_path}...')
    # teacher_model = get_teacher_model(args)
    # teacher_model = 0
    teacher_model =LlamaForCausalLM.from_pretrained(
        args.teacher_model_name_or_path,
        # device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        # teacher_model = teacher_model
    )
    # teacher_model =llama_two_model.Transformer()
    # weight = torch.load(f"{args.teacher_model_name_or_path}/pytorch_model.bin")
    # teacher_model.load_state_dict(weight)
    return teacher_model


def get_model(args):
    print(f'loading base model {args.model_name_or_path}...')
    # teacher_model = get_teacher_model(args)
    # print(teacher_model)
    # teacher_model = 0
    model = LlamaForCausalLM_stu.from_pretrained(
        args.model_name_or_path,
        # device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        # teacher_model = teacher_model
    )
    weight_origin = torch.load(args.teacher_model_name_or_path + "pytorch_model.bin")
    return LightningLlamaModel(model, weight_origin, args, max_seq_len=args.max_seq_len)

class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer

    def __call__(self, source_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        source_ids = torch.cat([source_id.unsqueeze(0) for source_id in source_ids], dim=0)
        return {
            'input_ids': source_ids,
            'attention_mask': source_ids.ne(0).long(), 
            'labels': source_ids,
        }
import random
# # 设置随机种子
# seed=2613
# torch.manual_seed(seed)
class DataModule(L.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
    
    def setup(self, stage=None):
        self.data_collator = DataCollatorForCausalLM(tokenizer=self.tokenizer)
        # SEED = 2613
        # random.seed(SEED)
        # filenames1 = glob.glob(f"{self.args.train_data_file}/slim_star_combined/*.bin")
        # random.shuffle(filenames1)
        filenames = glob.glob(f"{self.args.train_data_file}/slimpajama_combined1/*.bin")
        # filenames = filenames1[: int(len(filenames2) * 3 / 7)] + filenames2
        random.shuffle(filenames)
        # print(f"n_chunks == {self.args.n_chunks}")  # 8
        # print(f"block_size == {self.args.block_size}")  # 2048
        # print(f"seed == {self.args.seed}")  # 42
        self.dataset = PackedDataset(
            filenames=filenames,
            n_chunks=self.args.n_chunks,
            block_size=self.args.block_size,
            seed=self.args.seed,
            shuffle=True,
            wrap=True,
            num_processes=1,
            process_rank=0,
        )
        self.train_ds, self.val_ds = self.dataset.train_test_split(n_test_files=1)
        # for i, data in enumerate(self.dataset):
        #     print(data.max())
    
        # raise "stop"
        

    def train_dataloader(self):

        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=args.num_workers,
        )
    
    def val_dataloader(self):
        return [torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=0,
        )]

import pickle
from torch.utils.data import Dataset, DataLoader
# 用于读取通过teacher model后的数据
class MoE_Dataset(Dataset):
    def __init__(self, file_path, mode) -> None:
        super(MoE_Dataset, self).__init__()
        # self.filenames = glob.glob(f"{file_path}/*.bin")
        # input_ids_list = []
        # teacher_labels_list = []
        # for filename in self.filenames:
        #     with open(filename, 'rb') as f:
        #         loaded_data = pickle.load(f)
        #         input_ids_list.append(loaded_data['input_ids'])
        #         # teacher_labels_list.append(loaded_data['teacher_labels'])
        # self.input_ids = torch.concat(input_ids_list, dim=0)
        file_paths = glob.glob(f"{file_path}*.pt")
        input_ids = []
        for path in file_paths:
            input_ids.append(torch.load(path))
        self.input_ids = torch.concat(input_ids, dim=0)
        # self.teacher_labels = torch.concat(teacher_labels_list, dim=0)
        self.length = self.input_ids.shape[0]
        self.mode = mode

    def __len__(self):
        if self.mode == "train":

            return int(self.length*0.999)
        else:
            return self.length-int(self.length*0.999)
    
    def __getitem__(self, index):
        if self.mode == "train":
            return {
                "input_ids":  self.input_ids[index, :],
                'attention_mask' : self.input_ids[index, :].ne(0).long(),
                "labels" : self.input_ids[index, :],
                # "teacher_labels" : self.teacher_labels[index, :]
            }
        else:
            return {
                "input_ids":  self.input_ids[self.length-index-1, :],
                'attention_mask' : self.input_ids[self.length-index-1, :].ne(0).long(),
                "labels" : self.input_ids[self.length-index-1, :],
                # "teacher_labels" : self.teacher_labels[self.length-index, :]
            }

class MoE_DataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_datasets = MoE_Dataset(args.train_data_file, "train")
        self.val_datasets = MoE_Dataset(args.train_data_file, "val")
        self.args = args
    

    
    def train_dataloader(self):
        return DataLoader(self.train_datasets, batch_size=self.args.batch_size, num_workers=args.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_datasets, batch_size=self.args.batch_size, num_workers=0)


def main(args):


    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # # Load the datamodule
    # dm = DataModule(args, tokenizer)
    dm = MoE_DataModule(args)

    # dm.setup()
    # print(dm.train_dataloader)
    # train_dataloader = dm.train_dataloader()
    # j = 0
    # for i, data in enumerate(train_dataloader):
    #     # print(i, data)
    #     # print(data["input_ids"].shape)
    #     # print(data["attention_mask"].shape)
    #     # print(data["labels"].shape)
    #     # break
    #     j += 1
    # print(j)

    # raise "stop"


#     # Load the model
#     teacher_model = get_teacher_model(args)
#     teacher_model.model.to("cuda")
#     teacher_model.lm_head.to("cuda")
#     # model = get_model(args)
#     print("start")
#     with torch.no_grad():
#         for i, data in enumerate(train_dataloader):
#             j = i //150
#             i = i % 150
#             print(i, j)
#             # data = truncate_batch(data, args.max_seq_len)  # 裁剪长度
#             output = teacher_model.model(            
#                 input_ids=data["input_ids"].to("cuda"),
                # attention_mask=data["attention_mask"].to("cuda"),
                # position_ids=None,
                # past_key_values=None,
                # inputs_embeds=None,
                # use_cache=None,
                # output_attentions=False,
                # output_hidden_states=False,
                # return_dict=False,
#                 )
#             # print(output[0].shape)  # torch.Size([1, 1024, 2048])
#             # print(output[1][0])  # len(output[1]) == 22
#             hidden_state = output[0]
#             # print(teacher_model.pretraining_tp)

            # if teacher_model.pretraining_tp > 1:
            #     lm_head_slices = teacher_model.lm_head.weight.split(teacher_model.vocab_size // teacher_model.pretraining_tp, dim=0)
            #     logits = [torch.nn.functional.linear(hidden_state, lm_head_slices[i]) for i in range(teacher_model.pretraining_tp)]
            #     logits = torch.cat(logits, dim=-1)
            # else:
            #     logits = teacher_model.lm_head(hidden_state)

#             # print(teacher_model.lm_head.weight)
#             logits = logits.float()
#             # print(logits.shape)
#             # max_values, max_indices = torch.max(logits, dim=2)
#             # print(f"max_indices == {max_indices}") # tensor([[29896,    13,    13,  ...,   591,   338,   694]])
#             # print(f"max_values == {max_values}") # tensor([[ 9.8050, 10.5067, 20.3292,  ..., 10.6804, 16.3523, 12.3193]]
#             # print(f"max_indices.shape == {max_indices.shape}")  # torch.Size([1, 1024])
#             # print(f"max_values.shape == {max_values.shape}")  # torch.Size([1, 1024])
#             temp_data = {}
#             temp_data["teacher_labels"] = logits
#             temp_data["input_ids"] = data['input_ids']
#             # 将字典保存为二进制文件
#             with open(f'/vepfs-sha/dongmh/teacher_model_output/starcoder/train_starcoder_{j}_{i}.bin', 'wb') as f:
#                 pickle.dump(temp_data, f)
#     print("end")
#     del teacher_model
#     # teacher model处理完毕
#     args.train_data_file = "/vepfs-sha/dongmh/teacher_model_output/starcoder/"
#     # Load the datamodule



    # dm = MoE_DataModule(args)
    model = get_model(args)

    # print(model)



    # raise "stop"
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.experiment_name,
    )
    devices = args.devices if args.devices is not None else args.n_gpus

    # del devices[0] # 用第一块GPU跑teacher model


    checkpoint_callback = ModelCheckpoint(
        dirpath=f"ckpts/{args.experiment_name}",
        monitor=f"val_loss",
        filename="llama-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )
    # print(args.max_epochs)
    trainer = L.Trainer(
        max_steps=-1,
        max_epochs=args.max_epochs,
        devices=devices,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        precision="bf16-true",
        logger=logger,
        val_check_interval=2000,
        # check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.grad_accumulation_steps,

    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--teacher_model_name_or_path', type=str, required=True)
    parser.add_argument('--train_data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, required=True, default=500)
    parser.add_argument('--n_chunks', type=int, required=True, default=8)
    parser.add_argument('--block_size', type=int, required=True, default=2048)
    parser.add_argument('--max_seq_len', type=int, required=True, default=2048)
    parser.add_argument('--seed', type=int, required=True, default=42)
    parser.add_argument('--n_gpus', type=int, required=True, default=1)
    parser.add_argument('--devices', type=list, required=False, default=None)
    parser.add_argument('--batch_size', type=int, required=True, default=1)
    parser.add_argument('--num_workers', type=int, required=True, default=64)
    parser.add_argument('--trust_remote_code', type=bool, required=True, default=False)
    parser.add_argument('--grad_accumulation_steps', type=int, required=True, default=16)
    args = parser.parse_args()
    print(args)
    main(args)
