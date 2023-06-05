主要环境：

  torch=1.8.1+cu111

  numpy=1.21.6

  fastNLP=0.6.0

  transformers=3.4.0

配置文件：config.py

训练脚本：model.py 命令：python train.py

测试脚本：test.py 命令：python test.py

测试集：\entity_extraction\test.json  说明：如果是txt文件，请使用\entity_extraction\data.py 脚本处理为JSON文件

模型保存路径：\out\ent_res_best.pkl

测试结果：ent_res.txt
