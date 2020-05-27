#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sys
import jieba
import string
import time
import fasttext
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')


class DataProcessor:  # 中文短信分词、清洗处理

    def file_cut_words(self, in_folder, out_file, mode='vec'):  # 文本文件分词，生成分词后的新文件，mode选择vec为不带标签，选择flc为带标签
        label_files = os.listdir(in_folder)  # 读取目录下对应的分类文件（文件名为类别）
        # print(label_files)
        for fl_name in label_files:
            label = '_'.join(fl_name.split('_')[1:])
            label_file = os.path.join(in_folder, fl_name)
            lf = open(label_file, 'r', encoding='UTF-8')
            for line in lf:
                if not line:
                    continue

                msg = line.strip()
                seg = self.cut_words(msg)
                if not seg:
                    continue
                with open(out_file, 'a+', encoding='UTF-8') as target:  # 每个信息分词后加上__label__类别名
                    if mode == 'vec':
                        target.write('%s\n' % seg)
                    else:
                        target.write('%s\t__label__%s\n' % (seg, label))

    def cut_words(self, sentence):  # 句子分词
        words = list(jieba.cut(sentence))
        if not words:
            words = ['']

        new_words = self.wash_sentence(words)

        return " ".join(new_words)

    @staticmethod
    def wash_sentence(sentence):  # 去停用词
        stopWord_path = "E:\\BaiduNetdiskDownload\\NLP数据集\\stop\\stopword.txt"  # 停用词路径

        with open(stopWord_path, 'r', encoding='UTF-8') as stoper:
            words = [i.strip() for i in stoper.readlines()]

        set_words = set(words)
        for n in range(len(sentence)-1, -1, -1):  # 倒序去重
            if sentence[n] in set_words:
                sentence.pop(n)

        return sentence


class FtModel:  # Fasttext 文本分类模型 训练、预测

    def __init__(self, label_prefix='__label__', dim=300, word_ngrams=3, loss='hs', min_count=10):
        self.label_prefix = label_prefix  # 训练文本的标签前缀
        self.dim = dim  # 词向量的维数
        self.word_ngrams = word_ngrams  # n-gram 默认1,2或以上更好
        self.loss = loss  # 默认ns,可选ns,hs,softmax
        self.min_count = min_count  # 最低出现的词频 默认5
        self.DP = DataProcessor()

    # 此版本fasttext为新版本，train_unsupervised()用来单独生成词向量
    # 详见https://fasttext.cc/blog/2019/06/25/blog-post.html#2-you-were-using-the-unofficial-fasttext-module
    def w2v_train(self, documents_input, w2v_model_output):  # 预训练词向量并保存
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+' : create word-segment without label txt')
        documents_cut = 'cache/msg_seg_without_label.txt'
        self.DP.file_cut_words(documents_input, documents_cut, mode='vec')

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' : w2v train start')
        # skipgram模型训练生成词向量，结果输出到w2v_model_output：lr学习率，dim维数，min_count最小词频
        model = fasttext.train_unsupervised(documents_cut, model='skipgram', lr=0.05, dim=self.dim, loss=self.loss,
                                            word_ngrams=self.word_ngrams, min_count=self.min_count)
        model.save_model(w2v_model_output)

        # os.remove(documents_cut)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+' : w2v train done')

        return model

    # train_supervised()直接生成模型，带词向量模型和文本分类模型
    def ft_train(self, documents_input, ft_model_output, documents_cut='cache/msg_seg_with_label.txt'):  # Fasttext文本分类模型训练
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' : create word-segment with label txt')
        self.DP.file_cut_words(documents_input, documents_cut, mode='flc')

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' : ft train start')
        # fasttext文本分类训练，结果输出到ft_model_output：label_prefix标签前缀，pretrained_vectors词向量文件(可省略此参数)，dim维数
        classifier = fasttext.train_supervised(documents_cut, label=self.label_prefix, loss=self.loss, epoch=15,
                                               min_count=self.min_count, word_ngrams=self.word_ngrams, dim=self.dim)
        classifier.save_model(ft_model_output)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+' : ft train done')

        return classifier

    def load_model(self, ft_model_output):  # 加载模型

        return fasttext.load_model(ft_model_output)

    # 单句输入
    def text_predict(self, classifier, texts, is_test=0):  # 预测文本分类
        label_prefix = classifier.labels
        label_real = []
        texts_input = []
        text_col = 0

        if is_test:  # 是否为测试数据(测试数据多一列标签)
            text_col = 1
            label_real = [text[0] for text in texts]

        for text in texts:
            text = self.DP.cut_words(text[text_col])
            texts_input.append(text)


        predict_result = classifier.predict(texts_input)
        labels_predict = predict_result[0]
        # texts_output：第[0]个元素为文本，第[1]个元素为预测的标签结果
        texts_output = [[text, labels_predict[i][0]] for i, text in enumerate(texts_input)]

        # 如果为测试数据，texts_output追加第[2]个元素为测试数据的真实标签
        if is_test:
            for i, text in enumerate(label_real):
                texts_output[i].append(label_real[i])

        # [['信息内容', '预测标签', '真实标签'], [], [], ...]
        return texts_output


class FtTest:  # Fasttext 模型测试

    def __init__(self, classifier, infile):
        Model = FtModel()
        self.text = self.read_data(infile)  # 读取测试数据
        self.texts = Model.text_predict(classifier, self.text, 1)  # 预测分类

    def my_summary(self):
        texts = self.texts
        # print(texts[:2])
        predicted = [i[1] for i in texts]
        # print(predicted[:2])
        real = [i[2] for i in texts]
        # print(real[:2])

        print("精度：{0:.4f}".format(precision_score(real, predicted, average='weighted')))
        print("召回：{0:.4f}".format(recall_score(real, predicted, average='weighted')))
        print("f1-score:{0:.4f}".format(f1_score(real, predicted, average='weighted')))
        target_names = ['bank_xyk', 'bank_io', 'online_jy', 'cache', 'credit_ad',
                        'mobile_ll', 'mobile_hf', 'nonbank_dk', 'bank_dk']

        # label_names = ['__label__' + i + '.txt' for i in target_names]
        # print(classification_report(real, predicted, labels=label_names, target_names=target_names, digits=4))
        # reallable_ad = [i for i in texts if i[2] == '__label__credit_ad.txt']
        #
        # corect_p = [i for i in reallable_ad if i[1] == i[2]]
        # error_p = [i for i in reallable_ad if i[1] != i[2]]
        # print(len(corect_p))
        # print(len(error_p))
        # # print('Correct predict:')
        # # for i in corect_p:
        # #     print(i[0], '\n', 'True : ', i[2], '\n', 'Predict : ', i[1])
        # #     print('***************' * 5)
        # # print('#############' * 5)
        # print('Error predict:')
        # for i in error_p:
        #     print(i[0], '\n', 'True : ', i[2], '\n', 'Predict : ', i[1])
        #     print('***************' * 5)

        error_ad = [i for i in texts if all([i[1] == '__label__credit_ad.txt', i[1] != i[2]])]
        print(len(error_ad))
        print('Error predict:')
        dd = {}
        for i in error_ad:
            if i[2] not in dd:
                dd[i[2]] = 1
            else:
                dd[i[2]] += 1

        for k, v in dd.items():
            print(k, ":", v)
            # print(i[0], '\n', 'True : ', i[2], '\n', 'Predict : ', i[1])
            # print('***************' * 5)

    def summary(self, print_out=1):  # 统计准确率、召回率
        texts = self.texts
        lables_dict = {}  # 记录标签及序号
        summary = {}  # 记录各标签准确率、召回率
        print(len(texts))
        print(texts[0])
        for e in texts:
            lables_dict[e[1]] = 0  # 预测的标签
            lables_dict[e[2]] = 1  # 实际的标签

        lables_ls = lables_dict.keys()  # 标签列表
        print(lables_ls)
        lables_list = list(lables_ls)[0]
        confusion_matrix = []  # 混淆矩阵
        dim = len(lables_list)  # 标签个数
        for i, label in enumerate(lables_list):
            vector = [0 for _ in range(dim)]
            confusion_matrix.append(vector)
            lables_dict[label] = i  # 标签的序号
            summary[label] = [0, 0, 0]  # 每个标签的[实际数量,预测数量,预测与实际相符数量]

        for e in texts:
            label_predict = e[1]  # 预测标签
            label_real = e[2]  # 实际标签
            line_no = lables_dict[label_real]  # 实际标签的序号
            col_no = lables_dict[label_predict]  # 预测标签的序号
            confusion_matrix[line_no][col_no] += 1  # 混淆矩阵计数

        # 统计每个标签的[实际数量,预测数量,预测与实际相符数量]
        for line_no, vector in enumerate(confusion_matrix):
            label_real = lables_list[line_no]
            for col_no, cnt in enumerate(vector):
                label_predict = lables_list[col_no]
                summary[label_real][0] += cnt  # 实际标签的数量
                summary[label_predict][1] += cnt  # 预测标签的数量
                if line_no == col_no:  # 实际标签的序号 与 预测标签的序号 相同，即预测准确
                    summary[label_real][2] += cnt
                # print line_no,col_no,label_real,label_predict,cnt

        # 输出每个标签的标签名、样本数、准确率、召回率
        if print_out:
            print("label\tsupport\tprecision\trecall")
            for label in lables_list:
                support = summary[label][0]
                precision = '-'
                _ = summary[label][2]
                if support > 0:
                    precision = str(float(summary[label][2])/support)
                recall = '-'
                if summary[label][1] > 0:
                    recall = str(float(summary[label][2])/summary[label][1])
                # print label,support,precision,recall
                print("%s\t%s\t%s\t%s" % (label, support, precision, recall))

        return confusion_matrix, summary

    @staticmethod
    def read_data(infile):  # 读取测试文本数据
        fin = open(infile, encoding='UTF-8')
        texts = []
        for line in fin.readlines():
            line = line.strip()
            if not line or len(line) < 1:
                continue
            ln = line.split('__label')
            msg = ln[0]
            label = '__label' + ln[1]

            texts.append([label, msg])

        return texts


if __name__ == '__main__':

    train_data_path = 'D:/document/短信分类/msg-classfy-data/4-5-6-month/train/'
    test_data_path = 'D:/document/短信分类/msg-classfy-data/4-days/split-data/test-data/'
    w2v_model_output = 'model/model.vec'
    ft_model_output = 'model/model_%s.bin' % 'v3'

    Model = FtModel()
    # 训练词向量
    # Model.w2v_train(documents_input=train_data_path, w2v_model_output=w2v_model_output)
    # 训练文本分类模型
    # Model.ft_train(documents_input=train_data_path, ft_model_output=ft_model_output, documents_cut='cache/msg_seg_with_label_v3.txt')

    # 测试文件分级加标签
    # dp = DataProcessor()
    # dp.file_cut_words(test_data_path, 'cache/test_seg_with_label_v2.txt', mode='flc')

    # 测试分类效果，输出准确率、召回率
    testfile = 'cache/test_seg_with_label_v2.txt'
    classifier = Model.load_model(ft_model_output)
    Test = FtTest(classifier, testfile)
    Test.summary()

    # 预测分类结果,第[0]个元素为文本，第[1]个元素为预测的标签结果
    # classifier = fasttext.load_model(ft_model_output)
    # # # 模式0不知道标签
    # # # predict_result = Model.text_predict(classifier, [['我是哪一类的文本？'], ['我欠你钱吗'], ['小猪炳坤']], 0)
    # # # 模式1知道标签
    # predict_result = Model.text_predict(classifier, [['cache', '我是哪一类的文本？'], ['cache', '我欠你钱吗'], ['cache', '小猪炳坤']], 1)
    # print(predict_result)


    # sentence = '；？人民啊阿哎哎呀哎哟唉俺俺们按按照吧吧哒把罢了被本'
    # print(list(jieba.cut(sentence)))
    # print(DataProcessor().cut_words(sentence))

    # model = fasttext.load_model(ft_model_output)
    # print(model.words)
    # print(len(model['余额']))
    # print(model.labels)

    # testfile = 'cache/test_seg_with_label_v1.txt'
    # classifier = Model.load_model(ft_model_output)
    # Test = FtTest(classifier, testfile)
    # Test.my_summary()


