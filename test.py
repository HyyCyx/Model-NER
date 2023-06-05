import torch
from config import Config
from model import MatixModel,evaluate
from data import getdata_bundle,get_data_iterator
seed = 226
torch.manual_seed(seed)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__=="__main__":
    con = Config()
    data_bundle = getdata_bundle(con.train_datapath,con.dev_datapath,con.test_datapath)
    #data_bundle = getdata_bundle(con.train_datapath, con.test_datapath)
    test_dataset = data_bundle.get_dataset("test")
    test = get_data_iterator(con,test_dataset,istest=True)
    model = MatixModel(con).to(device)
    #训练完成的模型保存路径
    model.load_state_dict(torch.load(r"/home/dell/Model-NER/out/ent_res_best.pkl"))
    id2rel={id:rel for rel,id in con.ent2id.items()}
    prection,recall,f1=evaluate(model,test,id2rel,con)
    print("\nprection:{},recall:{},f1:{}\n".format(prection,recall,f1))

