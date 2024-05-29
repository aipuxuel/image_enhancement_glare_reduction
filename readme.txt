      此代码为推理代码，训练后的权重文件已放在snap文件夹内：glare.pth。在data/result/A文件夹内已有一些测试结果示例。

一、运行环境
      运行环境为一般的环境，出现no module named ***等问题，pip install对应的包即可。
二、数据格式
      将测试图片放在data/test_data/A，同时需要在data/result目录下创建同名的文件夹，例如A，以保存推理结果。
三、推理
      满足以上要求后，运行lowlight_test.py即可，运行结束后您可以在data/result/A文件夹内查看结果。