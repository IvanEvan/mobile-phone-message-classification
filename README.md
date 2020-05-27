## 简介

该项目实现了对用户手机短信的分类

具体类别应按照具体业务划分

------

## 模型简单对比

|             | 标注数据量 | 泛化能力 |
| ----------- | ---------- | -------- |
| naive-bayes | 较少       | 一般     |
| fast-text   | 较少       | 较好     |
| bert        | 较多       | 较强     |

------

## 鸣谢

Bert 采用 Google  ALBERT 的 tiny 版本预训练权重

代码采用苏神的 [bert4keras](https://github.com/bojone/bert4keras)

感谢他们的开源