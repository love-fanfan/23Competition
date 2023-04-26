# 23Competition

将23所提供的数据手动分为`TEST_DATA`和`TRAIN_DATA`。
其中，`TRAIN_DATA`中包括`frame_1.mat`到`frame_200.mat`。
其中，`TEST_DATA`中包括`frame_201.mat`到`frame_250.mat`。

执行`loadmat.m`，分别生成`test`和`train`，对应于训练集测试集。
执行：
```bash
python train.py
```
训练
