import json
class Config:
    def __init__(self):
        self.bertpath="/home/dell/Model-NER/sikuRoberta"
        self.bert_dim=768
        self.max_len=200
        #实体类型
        self.ent_num=len(eval(open("/home/dell/Model-NER/entity_extraction/ent2id.json").read()))
        self.ent2id=eval(open("/home/dell/Model-NER/entity_extraction/ent2id.json").read())
        #训练集
        self.train_datapath="/home/dell/Model-NER/entity_extraction/train.json"
        #验证集
        self.dev_datapath = "/home/dell/Model-NER/entity_extraction/dev.json"
        #测试集
        self.test_datapath="/home/dell/Model-NER/entity_extraction/test.json"
        self.batch_size=8
        self.val_batch_size=6
        self.res_savepath="res.txt"
        #测试结果
        self.valid_res="ent_res.txt"
        #训练好的模型保存路径
        self.model_savepath="/home/dell/Model-NER/out"
        self.lr=5e-5
        self.epoch=100
