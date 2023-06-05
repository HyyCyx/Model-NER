from transformers import BertModel,BertTokenizer,AutoModel,AutoTokenizer
import torch.nn as nn
import torch

import torch.nn.functional as F
from config import Config
from torch.optim import Adam

from data import getdata_bundle,get_data_iterator
from fastNLP import Trainer,LossBase,Callback
import json
from tqdm import tqdm
seed = 226
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NoneLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NoneLinear, self).__init__()
        self.linear = nn.Linear(input_dim, int(input_dim / 2))
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(int(input_dim / 2), output_dim)

    def forward(self, embed):
        output = self.linear(embed)
        feature = self.activate(output)
        feature = self.dropout(feature)
        return self.output(feature)

class MatixModel(nn.Module):
    def __init__(self,config):
        super(MatixModel,self).__init__()
        self.config=config
        self.bert=AutoModel.from_pretrained(self.config.bertpath)
        self.h_type=NoneLinear(self.config.bert_dim*2,self.config.ent_num*2)
        self.t_type=NoneLinear(self.config.bert_dim*2,self.config.ent_num*2)
        self.h_t=NoneLinear(self.config.bert_dim*2,2)
        self.rel_linear = nn.Linear(1, self.config.bert_dim)
    def getEmbed(self,text,mask,rel_pos):
        output=self.bert(text, attention_mask=mask)[0]
        ent1 = output.unsqueeze(2).expand(-1, -1, output.size(1), -1)+self.rel_linear(rel_pos.unsqueeze(-1))
        ent2 = output.unsqueeze(1).expand(-1, output.size(1), -1, -1)+self.rel_linear(rel_pos.unsqueeze(-1))
        et1_et2 = torch.cat((ent1, ent2), dim=3)
        return et1_et2,torch.cat((output,output),dim=-1)
    def getS_O_R(self,embed):

        pred_h_type=self.h_type(embed).reshape(embed.size(0), embed.size(1), self.config.ent_num, 2)
        pred_t_type=self.t_type(embed).reshape(embed.size(0), embed.size(1), self.config.ent_num, 2)
        return pred_h_type,pred_t_type
    def getH_T(self,embed):
        head_tail_matix=self.h_t(embed).reshape(embed.size(0), embed.size(1), embed.size(1), 2)
        return head_tail_matix
    def forward(self,input_ids,attention_mask,rel_pos):
        et1_et2,e2e=self.getEmbed(input_ids,attention_mask,rel_pos)
        pred_h_type,pred_t_type=self.getS_O_R(e2e)
        h_t=self.getH_T(et1_et2)
        return {
                "h_type":pred_h_type,
                "t_type":pred_t_type,
                "head_tail":h_t
                }
class MyLoss(LossBase):
    def __call__(self, pred_dict, target_dict, check=False):
        sub_obj_r_h = self.get_loss(pred_dict["h_type"], target_dict["ent_type_h"], target_dict["rel_mask"])
        sub_obj_r_t = self.get_loss(pred_dict["t_type"], target_dict["ent_type_t"], target_dict["rel_mask"])
        head_tail = self.get_loss(pred_dict["head_tail"], target_dict["head_tail"], target_dict["mask"])
        total_loss=0.5*(sub_obj_r_h + sub_obj_r_t)+ head_tail
        return total_loss

    def get_loss(self, pred, target, mask):
        pred=pred.permute(0, 3, 1, 2)
        loss = nn.CrossEntropyLoss(reduction="none")(pred, target)

        loss=torch.sum(loss*mask)/torch.sum(mask)
        return loss

def evaluate(model,dataset,id2type,con):
    file=open(con.valid_res,"w",encoding="utf-8")
    tokenizer=AutoTokenizer.from_pretrained(con.bertpath)
    with torch.no_grad():
        predicts,gold,true=0,0,0
        for batch in tqdm(dataset):
            inputs,target=batch[0],batch[1]
            sentence,attention_mask,rel_pos,input_sentence=inputs['input_ids'],inputs['attention_mask'], inputs['rel_pos'],inputs["input_sentence"]
            embed,e2e=model.getEmbed(sentence,attention_mask,rel_pos)
            pred_h_type,pred_t_type=model.getS_O_R(e2e)
            pred_h_type, pred_t_type = torch.argmax(pred_h_type, dim=-1), torch.argmax(pred_t_type, dim=-1)
            pre_h_t = torch.argmax(model.getH_T(embed), dim=-1)
            for idx,data in enumerate(inputs['input_ids']):
                inp=input_sentence[idx]

                spo_list=[(''.join(tokenizer.tokenize(str(d["ent"]))),str(d["ent_type"])) for d in target["ent_list"][idx]]
                sentence_id_len=len([i for i in inp if i!="[PAD]"])
                
                #subject_obj_h,sub_object_t
                hs,h_types=torch.where(pred_h_type[idx]==1)
                ts,t_types=torch.where(pred_t_type[idx]==1)
                #head,tail
                tails,heads=torch.where(pre_h_t[idx]==1)
                
                head_types=[]
                tail_types = []

                h_t=[]
                for index,i in enumerate(hs):
                    if i.item()<sentence_id_len and i.item()>0:
                            head_types.append((i.item(),h_types[index].item()))
                for index,i in enumerate(ts):
                    if i.item() < sentence_id_len and i.item() > 0:
                        tail_types.append((i.item(), t_types[index].item()))
                for index,i in enumerate(heads):
                    if i.item()<sentence_id_len and tails[index].item()<sentence_id_len and i.item()<=tails[index].item():
                        if i.item()>0 and tails[index].item()>0:
                            h_t.append((i.item(),tails[index].item()))
                #获得(s_h,s_t,o_h,o_t,r)对
                subject_rel_objects=[]
                subject_type_objects=[]
                ents_indexs=[]
                for h_type in head_types:
                    for t_type in tail_types:
                        if h_type[1]==t_type[1]:
                            if (h_type[0],t_type[0]) in h_t:
                                ent=''.join(inp[h_type[0]-1:t_type[0]])
                                type=id2type[h_type[1]]
                                ents_indexs.append((h_type[0]-1,t_type[0]-1))
                                subject_rel_objects.append((ent,type))
                                subject_type_objects.append((ent,type,(h_type[0]-1,t_type[0]-1)))
                predicts+=len(set(subject_rel_objects))
                cut_indexs=[]
                for sub_index in subject_type_objects:
                    for ent_index in ents_indexs:
                        if (sub_index[2][0]>=ent_index[0] and sub_index[2][1]<ent_index[1]) or sub_index[2][0]>ent_index[0] and sub_index[2][1]<=ent_index[1]:
                            cut_indexs.append(sub_index)
                subject_type_objects=[i for i in subject_type_objects if i not in cut_indexs]
                gold+=len(set(spo_list))
                true+=len(set(spo_list)&set(subject_rel_objects))
                pre_spo_list=[]
                gold_spo_list=[]
                for s_o_r in set(subject_type_objects):
                    pre_spo_list.append({"ent":s_o_r[0],"ent_type":s_o_r[1],"ent_span":s_o_r[2]})
                    inp[s_o_r[2][0]]="{"+inp[s_o_r[2][0]]
                    inp[s_o_r[2][1]]=inp[s_o_r[2][1]]+'|'+s_o_r[1]+'}'
                for d in spo_list:
                    gold_spo_list.append({"ent":d[0],"ent_type":d[1]})
                file.write(''.join([i for i in inp if i!="[PAD]" and i!="[SEP]" and i!="[CLS]"])+'\n')
        prection=true/predicts if predicts > 0 else 0
        recall=true/gold if gold > 0 else 0
        f1=2*prection*recall/(prection+recall) if (prection+recall) > 0 else 0
        return prection,recall,f1

class MyCallBack(Callback):
    def __init__(self,train_dataset,data,con,test_data,optimizer):
        super(MyCallBack,self).__init__()
        self.dev_data=data
        self.con=con
        self.file=open(self.con.res_savepath,'w',encoding='utf-8')
        self.best_f1=0
        self.best_epoch=0
        self.test_data=test_data
        self.scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-6,max_lr=5e-5,step_size_up=25,mode="triangular2",cycle_momentum=False)
    def on_epoch_end(self):
        id2rel={id:rel for rel,id in self.con.ent2id.items()}
        prection,recall,f1=evaluate(self.model,self.dev_data,id2rel,self.con)
        self.scheduler.step()
        if f1>=self.best_f1 and f1!=0:
            self.best_f1=f1
            self.best_epoch=self.epoch
            test_prection,test_recall,test_f1=evaluate(self.model,self.test_data,id2rel,self.con)
            self.file.write("epoch:{},test:prection:{},recall:{},f1:{}\nval:prection:{},recall:{},f1:{}\n".format(self.epoch,test_prection,test_recall,test_f1,prection,recall,f1))
            print(
                "epoch:{},test:prection:{},recall:{},f1:{}\nval:prection:{},recall:{},f1:{}\n".format(self.epoch,
                                                                                                      test_prection,
                                                                                                      test_recall,
                                                                                                      test_f1, prection,
                                                                                                      recall, f1))

            torch.save(self.model.state_dict(),self.con.model_savepath+'/ent_res.pkl')
if __name__=="__main__":
    con=Config()
    data_bundle=getdata_bundle(con.train_datapath,con.dev_datapath,con.test_datapath)

    train_dataset,dev_dataset,test_dataset=data_bundle.get_dataset("train"),data_bundle.get_dataset("dev"),data_bundle.get_dataset("test")
    train,dev,test=get_data_iterator(con,train_dataset),get_data_iterator(con,dev_dataset),get_data_iterator(con,test_dataset,istest=True)
    model=MatixModel(con)
    if torch.cuda.device_count()>0:
        model=nn.DataParallel(model)
        model=model.module
    else:model=model
    model.to(device)
    #optimizer=Adam(optimizer_grouped_parameters)
    optimizer=Adam(model.parameters(),lr=con.lr)
    trainer = Trainer(train_data=train, model=model, optimizer=optimizer, batch_size=con.batch_size,
                      n_epochs=con.epoch, loss=MyLoss(), use_tqdm=True,
                      callbacks=MyCallBack(train,dev,con,test,optimizer))
    trainer.train()
