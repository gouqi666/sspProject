### 说明
>  - 本项目分为SSP(Speech Signal Processing)大作业。主要模型结构：Wakeup + AM + LM  + Dilogue Policy + TTS。
>   - Wakeup是用的 SPECT + CNN，用的时候自己将训练模型放置在model目录下，再predict或api文件中调用即可。
>  - AM 是用的tensorflow，必须是tensorflow2.2.0,模型用的是 Conformer + CTC loss
>  - LM 用的transformer encoder，输入是AM模型输出的pinyin序列，输出结果是汉字。

----
### 目录构成

> - financeApi：金融数据调取，message生成；用于与chatbot交互
> - SASR2：最终的语音对话助手

### 运行
运行之前：需要有如下模型：
- wakeupModule模型（可以运行myWakeup\train_data的add_train_data.py脚本来增加训练数据，然后运行myWakeup\trainer.py来训练唤醒模型）
- AM的模型（到对应链接去下载[AM模型](pan.baidu.com/s/1NPk17DUr0-lBgwCkC5dFuQ)(密码：7qmd)放置在conformerCTC(M)中即可）
- LM的模型（可以自己训练，数据集用的是aishell,文本可以用pypinyin转化）。
> 1. 先运行financeApi下的market.py文件
>
> 2. 配置chatbot上的api调用的ip地址
>
> 3. 运行SASR2下的main.py文件  



### 环境

> tensorflow 2.2.0
>
> torch 1.10.0
>
> ......

### 参考

https://github.com/yeyupiaoling/MASR  
https://github.com/Z-yq/TensorflowASR