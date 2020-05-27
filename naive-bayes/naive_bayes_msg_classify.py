#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/7 15:23
# @Author  : Evan
# @File    : naiveBayes-textClassification.py
# 导入第三方库
import jieba
from numpy import *
import pickle  # 持久化
import os
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法
from sklearn import metrics
# from sklearn.externals import joblib
import joblib
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')


# 读取数据
def readFile(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        file.close()
        return content


def readFiles(path):
    with open(path, 'r', encoding='UTF-8') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = [i.strip() for i in file.readlines()]
        file.close()
        return content


# 写入数据
def saveFile(path, result):
    with open(path, 'w', errors='ignore') as file:
        file.write(result)
        file.close()


def saveFiles(path, result):
    with open(path, 'w', encoding='UTF-8') as file:
        for i in result:
            file.write(i + '\n')
        file.close()


# 对原始数据进行清洗并分词/保存分词后的结果
def segText(inputPath, resultPath):
    label_list = os.listdir(inputPath)  # 主目录

    if not os.path.exists(resultPath):  # 是否存在，不存在则创建
        os.makedirs(resultPath)

    for file_name in label_list:
        label_file = os.path.join(inputPath, file_name)
        content_ls = readFiles(label_file)
        result = [(str(i)).replace("\r\n", "").strip() for i in content_ls]  # 删除多余空行与空格
        cutResult = [" ".join(jieba.cut(i)) for i in result]  # 默认方式分词，分词结果用空格隔开
        saveFiles(os.path.join(resultPath, file_name), cutResult)  # 调用上面函数保存文件


def bunchSave(inputFile, outputFile, mode='train'):
    catelist = os.listdir(inputFile)
    if mode == 'train':
        label = [i.split('.txt')[0].split('train_')[1] for i in catelist]
    else:
        label = [i.split('.txt')[0].split('test_')[1] for i in catelist]

    bunch = Bunch(target_name=[], label=[], contents=[])
    bunch.target_name.extend(label)  # 将类别保存到Bunch对象中
    for k, eachDir in enumerate(catelist):

        label_file = os.path.join(inputFile, eachDir)
        content_ls = readFiles(label_file)
        for j in content_ls:
            bunch.label.append(label[k])  # 当前分类标签
            bunch.contents.append(j)  # 保存文件词向量
    with open(outputFile, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)


def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)


# 停用词
def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList


# 求TF-IDF向量
def getTFIDFMat(inputPath, stopWordList, outputPath):
    bunch = readBunch(inputPath)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, contents=bunch.contents, tdm=[], vocabulary={})
    # 初始化向量空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    # transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_  # 获取词汇
    writeBunch(outputPath, tfidfspace)
    # print(tfidfspace.tdm.todense().shape)


def getTestSpace(testSetPath, trainSpacePath, stopWordList, testSpacePath):
    bunch = readBunch(testSetPath)
    # 构建测试集TF-IDF向量空间
    testSpace = Bunch(target_name=bunch.target_name, label=bunch.label, contents=bunch.contents, tdm=[], vocabulary={})
    # 导入训练集的词袋
    trainbunch = readBunch(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    # transformer = TfidfTransformer()
    print(bunch.contents)
    testSpace.tdm = vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary = trainbunch.vocabulary
    # 持久化
    writeBunch(testSpacePath, testSpace)


def train_model(datapath, split_datapath, train_dat_path, stopWord_path, tfidfspace_dat_path):
    # 输入训练集
    segText(datapath,  # 读入数据
            split_datapath  # 输出分词结果
            )

    bunchSave(split_datapath,  # 读入分词结果
              train_dat_path,  # 输出分词向量
              mode='train'
              )

    stopWordList = getStopWord(stopWord_path)  # 获取停用词表

    getTFIDFMat(train_dat_path,  # 读入分词的词向量
                stopWordList,    # 获取停用词表
                tfidfspace_dat_path  # tf-idf词频空间向量的dat文件
                )
    trainSet = readBunch(tfidfspace_dat_path)

    clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)

    if not os.path.exists('model'):  # 是否存在，不存在则创建
        os.makedirs('model')

    joblib.dump(clf, 'model/bayes-model.pkl')


def predict_one_msg(massage, trainSpacePath):
    msg = " ".join(list(jieba.cut(massage)))
    # 导入训练集的词袋
    trainbunch = readBunch(trainSpacePath)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainbunch.vocabulary)
    # transformer = TfidfTransformer()
    msg_tdm = vectorizer.fit_transform([msg])

    clf = joblib.load('model/bayes-model.pkl')

    predicted = clf.predict(msg_tdm)

    return predicted


def bayesAlgorithm(trainPath, testPath):
    trainSet = readBunch(trainPath)
    testSet = readBunch(testPath)
    # print(trainSet.tdm.shape, len(trainSet.label))
    clf = joblib.load('model/bayes-model.pkl')
    # clf = MultinomialNB(alpha=0.001).fit(trainSet.tdm, trainSet.label)
    # alpha:0.001 alpha 越小，迭代次数越多，精度越高
    # print(shape(trainSet.tdm)) # 输出单词矩阵的类型
    # print(shape(testSet.tdm))
    predicted = clf.predict(testSet.tdm)
    # predicted = clf.predict_proba(testSet.tdm)
    # print(predicted)
    total = len(predicted)
    rate = 0
    # for flabel, msg, expct_cate in zip(testSet.label, testSet.contents, predicted):
    #     if flabel != expct_cate:
    #         rate += 1
    #         print('@@' * 20)
    #         print(msg, '\n', "实际类别：", flabel, '\n', "预测类别：", expct_cate)
    # print("erroe rate:", float(rate) * 100 / float(total), "%")
    # print("精度：{0:.4f}".format(precision_score(testSet.label, predicted, average='weighted')))
    # print("召回：{0:0.4f}".format(recall_score(testSet.label, predicted, average='weighted')))
    # print("f1-score:{0:.4f}".format(f1_score(testSet.label, predicted, average='weighted')))
    # print("精度：{}".format(precision_score(testSet.label, predicted, average=None)))
    # print("召回：{}".format(recall_score(testSet.label, predicted, average=None)))
    # print("f1-score:{}".format(f1_score(testSet.label, predicted, average=None)))
    print(testSet.label)
    target_names = ['bank_xyk', 'bank_io', 'online_jy', 'cache', 'credit_ad', 'mobile_ll', 'mobile_hf', 'nonbank_dk',
                    'bank_dk']
    label_names = target_names
    print(classification_report(testSet.label, predicted, labels=label_names, target_names=target_names, digits=4))
    print("错误数量：", rate)
    print(len(testSet.contents))


if __name__ == '__main__':
    train_path = "D:\\document\\短信分类\\msg-classfy-data\\4-days\\split-data\\train-data\\"  # 原始数据路径
    stopWord_path = "E:\\BaiduNetdiskDownload\\NLP数据集\\stop\\stopword.txt"  # 停用词路径
    test_path = "D:\\document\\短信分类\\msg-classfy-data\\4-days\\split-data\\test-data\\"  # 测试集路径
    '''
    以上三个文件路径是已存在的文件路径，下面的文件是运行代码之后生成的文件路径
    dat文件是为了读取方便做的，txt文件是为了给大家展示做的，所以想查看分词，词频矩阵
    词向量的详细信息请查看txt文件，dat文件是通过正常方式打不开的
    '''
    train_split_path = os.sep.join(['.', 'split', 'train_split', os.sep])  # 训练集分词路径
    test_split_path = os.sep.join(['.', 'split', 'test_split', os.sep])  # 测试集分词路径
    '''
    以上两个路径是分词之后的文件路径，大家可以生成之后自行打开查阅学习
    '''
    train_dat_path = os.sep.join(['.', 'train_set.dat'])  # 读取分词数据之后的bunch保存为二进制文件
    tfidfspace_dat_path = os.sep.join(['.', 'tfidfspace.dat'])  # tf-idf词频空间向量的dat文件

    test_dat_path = os.sep.join(['.', 'test_set.dat'])  # 测试集分词bat文件路径
    testspace_dat_path = os.sep.join(['.', 'testspace.dat'])   # 测试集输出空间矩阵dat文件
    '''
    以上四个为dat文件路径，是为了存储信息做的，不要打开
    '''
    # 获取停用词表
    stopWordList = getStopWord(stopWord_path)

    # 输入训练集
    # segText(train_path,  # 读入数据
    #         train_split_path  # 输出分词结果
    #         )
    # bunchSave(train_split_path,  # 读入分词结果
    #           train_dat_path,  # 输出分词向量
    #           mode='train'
    #           )
    #
    # getTFIDFMat(train_dat_path,  # 读入分词的词向量
    #             stopWordList,    # 获取停用词表
    #             tfidfspace_dat_path  # tf-idf词频空间向量的dat文件
    #             )
    '''
    测试集的每个函数的参数信息请对照上面的各个信息，是基本相同的
    '''
    # 输入测试集
    # segText(test_path,
    #         test_split_path  # 对测试集读入文件，输出分词结果
    #         )
    # bunchSave(test_split_path,
    #           test_dat_path,
    #           mode='test'
    #           )
    # getTestSpace(test_dat_path,
    #              tfidfspace_dat_path,
    #              stopWordList,
    #              testspace_dat_path
    #              )  # 输入分词文件，停用词，词向量，输出特征空间(txt,dat文件都有)
    # bayesAlgorithm(tfidfspace_dat_path,
    #                testspace_dat_path
    #                )



    # train_model(train_path, train_split_path, train_dat_path, stopWord_path, tfidfspace_dat_path)
    p = predict_one_msg('退款提醒：中国建设银行银行卡(5054)预计入账时间2016-04-27 20:15，若超时未收到，请联系银行。【支付宝】', tfidfspace_dat_path)
    print(p)




