import json
file=open("/home/dell/Model-NER/entity_extraction/test.json","w",encoding="utf-8")
for line in open("/home/dell/Model-NER/entity_extraction/GuNER2023_test_public.txt").readlines():
    file.write(json.dumps({"text":line.strip(),"ent_list":[{"ent": 1,"ent_type":1}]},ensure_ascii=False)+'\n')
