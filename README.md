主要环境：
  python=3.7.0
  
  torch=1.8.1+cu111

  numpy=1.21.6

  fastNLP=0.6.0

  transformers=3.4.0
  
entity_extraction：用于存放训练集、验证集、测试集

out：用于存放训练好的模型

sikuRoberta：用于存放预训练模型

配置文件：config.py，训练前请先核对该文件内相关文件路径

训练脚本：model.py 命令：python train.py

测试脚本：test.py 命令：python test.py

测试集：\entity_extraction\test.json  说明：如果是txt文件，请先使用\entity_extraction\data.py 脚本，将txt文件处理为JSON文件

模型保存路径和保存的文件名：\out\ent_res_best.pkl

测试结果：ent_res.txt
