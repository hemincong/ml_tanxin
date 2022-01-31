# 方法1: 自己建立一个停用词词典
stop_words = ["the", "an", "is", "there"]
# 在使用时: 假设 word_list包含了文本里的单词
word_list = ["we", "are", "the", "students"]
filtered_words = [word for word in word_list if word not in stop_words]
print (filtered_words)
# 方法2:直接利用别人已经构建好的停用词库
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
print(cachedStopWords)