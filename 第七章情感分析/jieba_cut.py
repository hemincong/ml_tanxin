# encoding=utf-8
import jieba
# 基于jieba的分词 参考: https://github.com/fxsjy/jieba
seg_list = jieba.cut("贪心学院是国内最专业的人工智能在线教育品牌", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))
# 在jieba中加入"贪心学院"关键词
jieba.add_word("贪心学院")
seg_list = jieba.cut("贪心学院是国内最专业的人工智能在线教育品牌", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))
