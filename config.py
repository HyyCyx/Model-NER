import json
class Config:
    def __init__(self):
        self.bertpath="/home/dell/Model-NER/sikuRoberta"
        self.bert_dim=768
        self.max_len=200
        self.ent_num=len(eval(open("/home/dell/Model-NER/entity_extraction/ent2id.json").read()))
        self.ent2id=eval(open("/home/dell/Model-NER/entity_extraction/ent2id.json").read())
        self.train_datapath="/home/dell/Model-NER/entity_extraction/train.json"
        self.dev_datapath = "/home/dell/Model-NER/entity_extraction/dev.json"
        self.test_datapath="/home/dell/Model-NER/entity_extraction/test.json"
        self.batch_size=8
        self.val_batch_size=6
        self.res_savepath="res.txt"
        self.valid_res="ent_res.txt"
        self.model_savepath="/home/dell/Model-NER/out"
        self.lr=5e-5
        self.epoch=100
