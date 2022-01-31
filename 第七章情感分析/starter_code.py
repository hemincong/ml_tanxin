def process_file():
    """
    读取训练数据和测试数据，并对它们做一些预处理
    """
    train_pos_file = "data/train_positive.txt"
    train_neg_file = "data/train_negative.txt"
    test_comb_file = "data/test_combined.txt"

    # TODO: 读取文件部分，把具体的内容写入到变量里面
    train_comments = []
    train_labels = []
    test_comments = []
    test_labels = []

    from bs4 import BeautifulSoup
    pos_soup = BeautifulSoup(open(train_pos_file), features="lxml")
    reviews = pos_soup.find_all('review')

    for review in reviews:
        train_comments.append(review.get_text().strip())
        train_labels.append(1)

    neg_soup = BeautifulSoup(open(train_neg_file), features="lxml")
    reviews = neg_soup.find_all('review')
    for review in reviews:
        train_comments.append(review.get_text().strip())
        train_labels.append(0)

    test_soup = BeautifulSoup(open(test_comb_file), features="lxml")
    reviews = test_soup.find_all('review')
    for review in reviews:
        test_comments.append(review.get_text().strip())
        test_labels.append(review.get('label'))

    print (len(train_comments), len(test_comments))

process_file()
#print (len(train_comments), len(test_comments))

