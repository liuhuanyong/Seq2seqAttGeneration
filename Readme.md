# 项目的由来
1、分类、抽取、序列标注、生成任务是自然语言处理的四大经典任务，其中，分类、抽取任务，可以使用规则进行快速实现。而对于生成而言，则与统计深度学习关系较为密切。  
2、当前，GPT系列，自动文本生成、文本图像生成，图像文本生成等魔幻主义大作频频上演。  
3、目前开源的seq2seq模型项目晦涩难度，不利于阅读与入门。  
受此三个现实背景，也正好在接触生成这个任务，特做此项目。  

# 项目的构成
项目场景：该项目以自动对诗为使用场景，即用户给定上一句，要求模型给出下一句，是个较理想的生成例子。  
项目代码结构：  
    data/data.txt:为训练数据，此处使用的是对联诗句数据  
    att_seq2seq_predict.py:使用seq2seq模型进行下一句生成的脚本  
    seq2seq_train.py:使用seq2seq模型进行生成的脚本  
    model/:  
        cseq2seq_config.json:预训练时形成的一些关键参数，如最大长度、输入语句的字符索引等
        model.weights:训练好的模型  
 
# 项目的思想：
参考： https://kexue.fm/archives/5861
采用character字级别，通过搭建lstm-encoder和lstm-decoder和attention进行seq2seq生成任务。  

# 项目的使用：
1、python att_seq2seq_train.py,进行模型训练。  
2、python att_seq2seq_predict.py,进行模型测试。  

# 项目的总结：

1，本项目完成了一个基于keras实现的自动对诗文本生成功能。  
2，这是一个较为简单的入门级项目，欢迎补充，对基于https://kexue.fm/archives/5861进一步规整以更方便使用。  


