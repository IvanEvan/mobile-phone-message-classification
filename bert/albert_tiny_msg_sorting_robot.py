#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 10:12
# @Author  : Evan / Ethan
# @File    : albert_tiny_msg_sorting_robot.py
import os
import sys
import time
import numpy as np
import logging

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from keras.models import Model

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()

# config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


# roberta-tiny
# config_path = 'models/chinese_roberta_L-4_H-312_A-12/bert_config.json'
# checkpoint_path = 'roberta_best_model.weights'
# dict_path = 'models/chinese_roberta_L-4_H-312_A-12/vocab.txt'
# mode = 'bert'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class SubDataGenerator(DataGenerator):
    """数据生成器
    """
    def __init__(self, data, tokenizer, max_len, batch_size):
        super(SubDataGenerator, self).__init__(data, batch_size=batch_size, buffer_size=None)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids]
                batch_token_ids, batch_segment_ids = [], []


def load_data(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]

    data_ls = [i.split('\t')[-1] for i in lines]

    return data_ls


def predict(data, model, id2label):

    out_ls_id = []
    for x_true in data:
        # t1 = time.time()
        y_pred = list(model.predict(x_true).argmax(axis=1))
        # t2 = time.time()
        # logging.info('Predict data in %s s' % (str(t2 - t1)))
        out_ls_id += y_pred

    return list(map(lambda x: id2label[int(x)], out_ls_id))


def init_model(gpu_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    num_classes = 9

    # albert-tiny
    config_path = 'albert_tiny_zh_google/albert_config_tiny_g.json'
    my_checkpoint_path = 'albert_tiny_zh_google/albert_tiny_best_model.weights'
    dict_path = 'albert_tiny_zh_google/vocab.txt'
    mode = 'albert'

    tokenizer = Tokenizer(os.path.join(os.getcwd(), dict_path), do_lower_case=True)  # 建立分词器
    bert = build_transformer_model(config_path=os.path.join(os.getcwd(), config_path),
                                   checkpoint_path=None,
                                   model=mode,
                                   return_keras_model=False)  # 建立模型，加载权重

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    output = Dense(units=num_classes,
                   activation='softmax',
                   kernel_initializer=bert.initializer
                   )(output)

    model = Model(bert.model.input, output)
    model.summary()

    model.load_weights(os.path.join(os.getcwd(), my_checkpoint_path))

    return model, tokenizer


if __name__ == '__main__':
    gpu_number = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    maxlen = 512
    batch_size_num = 32

    zh_label = ['其他', '运营商_流量', '运营商_话费', '银行_信用卡', '银行_流水', '银行_贷款', '非银行_信贷广告', '非银行_消费', '非银行_贷款']
    dg_label = list(range(9))
    dg2zh_label = dict(zip(dg_label, zh_label))

    bert_model, bert_tokenizer = init_model(gpu_number)

    test_data = load_data(input_file)
    test_generator = SubDataGenerator(test_data, bert_tokenizer, maxlen, batch_size_num)
    # print(test_data)

    out_str = '\n'.join(predict(test_generator, bert_model, dg2zh_label))

    with open(output_file, 'w', encoding='utf-8') as fo:
        fo.write(out_str)
    #
    # for i in range(9):
    #     hit_num = 0
    #     total_num = 0
    #     for idx, lbl in enumerate(list(total_lbls)):
    #         if int(lbl) == i:
    #             total_num += 1
    #             if list(total_pres)[idx] == lbl:
    #                 hit_num += 1
    #
    #     print('Label %d | Correct %d | Total %d | Acc %.4f' % (i, hit_num, total_num, hit_num/total_num))
    #
    # print('Total acc %.4f' % tt_auc)
