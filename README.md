# Model-NER
环境: \n
    torch=1.8.1+cu111 \n
    numpy=1.21.6 \n
    fastNLP=0.6.0 \n
    transformers=3.4.0 \n
配置文件：config.py
测试文件：test.py
测试集：\entity_extraction\test.json
	说明：如果是txt文件，请使用\entity_extraction\data.py 脚本处理为JSON文件
模型保存路径：\out\ent_res_best.pkl
测试结果：ent_res.txt
