from fastNLP.io import JsonLoader
from fastNLP import Sampler, DataSet, TorchLoaderIter
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer,AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getdata_bundle(train_path,dev_path, test_path):
    loader = JsonLoader({"text": "text", "ent_list": "ent_list"})
    paths = {"train": train_path,"dev": dev_path, "test": test_path}
    data_bundle = loader.load(paths)
    return data_bundle


class Dataload(DataSet):
    def __init__(self, con, dataset) -> None:
        super().__init__()
        self.con = con
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.con.bertpath)

    def __getitem__(self, idx):
        input_sentence, spo_list = self.dataset[idx]['text'], self.dataset[idx]['ent_list']

        token = self.tokenizer(input_sentence, max_length=self.con.max_len, truncation=True)
        input_ids, attention_mask = torch.tensor(token["input_ids"], dtype=torch.long), torch.tensor(
            token['attention_mask'])
        tokenize = self.tokenizer.convert_ids_to_tokens(input_ids)
        sub_obj_rel_idx = []
        for spo in spo_list:
            ent, type= str(spo['ent']), str(spo['ent_type'])
            ent_token= self.tokenizer.tokenize(ent)
            head_idx_list= find_head_idx(tokenize, ent_token)
            if type=="1":
                rel_idx=0
            else:
                rel_idx = self.con.ent2id[type]

            if head_idx_list != -1:
                for head_idx in head_idx_list:
                    ent_tail_idx= head_idx + len(ent_token) - 1
                    sub_obj_rel_idx.append((head_idx, ent_tail_idx, rel_idx))
        pos = torch.tensor([i for i in range(len(token["input_ids"]))], dtype=torch.float32)
        return input_ids, attention_mask, spo_list, self.con, sub_obj_rel_idx,pos,list(input_sentence)

    def __len__(self):
        return len(self.dataset)


def find_head_idx(source, target):
    target_len = len(target)
    res=[]
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            res.append(i)
    if len(res)!=0:
        return res
    else:
        return -1


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    input_ids, attention_mask, spo_list, con, sub_obj_rel_idx,pos ,input_sentence= zip(*batch)
    batch_input_ids = pad_sequence(input_ids, batch_first=True)
    batch_mask = pad_sequence(attention_mask, batch_first=True)
    batch_pos=pad_sequence(pos,batch_first=True)
    rel_pos=(batch_pos.unsqueeze(2)).expand(batch_pos.size(0),batch_pos.size(1),batch_pos.size(1))-(batch_pos.unsqueeze(1)).expand(batch_pos.size(0),batch_pos.size(1),batch_pos.size(1))

    mask1 = (batch_mask.unsqueeze(2)).expand(-1, -1, batch_mask.size(1))
    mask = mask1 + mask1.transpose(1, 2)
    mask = (mask.masked_fill(mask == 1, 0)).masked_fill(mask == 2, 1)
    rel_pos=rel_pos.masked_fill(mask==0,0)
    rel_mask = (batch_mask.unsqueeze(-1)).expand(-1, -1, con[0].ent_num)
    batch_ent_type_h = torch.zeros((batch_input_ids.size(0), batch_mask.size(1), con[0].ent_num),dtype=torch.long)
    batch_ent_type_t = torch.zeros((batch_input_ids.size(0), batch_mask.size(1), con[0].ent_num),dtype=torch.long)
    batch_head_tail = torch.zeros((batch_mask.size(0), batch_mask.size(1), batch_mask.size(1)),dtype=torch.long)
    for i, spo_lists in enumerate(sub_obj_rel_idx):
        for spo in spo_lists:
            head_idx, tail_idx, type_idx= spo[0], spo[1], spo[2]
            batch_head_tail[i][tail_idx][head_idx] = 1
            batch_ent_type_h[i][head_idx][type_idx] = 1
            batch_ent_type_t[i][tail_idx][type_idx] = 1
    return {"input_ids": batch_input_ids.to(device), "attention_mask": batch_mask.to(device),"rel_pos": rel_pos.to(device),"input_sentence":input_sentence}, \
           {"ent_list": spo_list, \
            "ent_type_h": batch_ent_type_h.to(device), \
            "ent_type_t": batch_ent_type_t.to(device), \
            "head_tail": batch_head_tail.to(device), \
            "mask": mask.to(device), \
            "rel_mask": rel_mask.to(device)}


class MySample(Sampler):
    def __call__(self, data_set):
        return [i for i in range(len(data_set))]


def get_data_iterator(config, dataset, istest=False, collate_fn=collate_fn):
    dataload = Dataload(config, dataset)
    return TorchLoaderIter(dataload, collate_fn=collate_fn, \
                           batch_size=6 if istest == True else config.batch_size, sampler=MySample())


# if __name__ == "__main__":
#     con = Config()
#     data_bundle = getdata_bundle("/home/dell/Model_attention/data/train.json",
#                                  "/home/dell/Model_attention/data/dev.json",
#                                  "/home/dell/Model_attention/data/test.json")
#     dataset = data_bundle.get_dataset("train")
#     dataload = Dataload(con, dataset)
#     istest = True
#     res = TorchLoaderIter(dataload, collate_fn=collate_fn, batch_size=6 if istest == True else con.batch_size,
#                           sampler=MySample())
#     for r in res:
#         r
