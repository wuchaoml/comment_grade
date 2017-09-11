import jieba
import jieba.posseg as pseg
import chardet
import pickle


class ProductReviews(object):
    """docstring for ProductReviews"""

    def __init__(self, product_comment_file_name):
        super(ProductReviews, self).__init__()
        # 评论文件的名称
        self.product_comment_file_name = product_comment_file_name
        # 停止词列表
        self.stop_words_list = []
        # 分词和词性标注后的列表
        self.words_posseg = []
        # 商品评论的数量
        self.line_num = 0
        # 名词列表
        self.noun_list = []

    def stopWordsLoad(self, stop_words_name):
        '''加载停止词列表'''
        with open('%s.txt' % (stop_words_name), 'r') as f:
            while True:
                stop_word = f.readline().strip()
                if len(stop_word) != 0:
                    self.stop_words_list.append(stop_word)
                else:
                    break
        pickleDump('stop_words_list', self.stop_words_list)
        print('Stop words is OK!')

    def segSentence(self):
        '''分词以及词性标注'''
        with open('%s.txt' % (self.product_comment_file_name), 'r') as f:
            while True:
                line = f.readline()
                if len(line) != 0:
                    self.line_num += 1
                    # seg_list = jieba.cut(line, cut_all=False, HMM=True)
                    words = pseg.cut(line)
                    self.wordDeal(words)
                else:
                    pickleDump('words_posseg', self.words_posseg)
                    pickleDump('noun_list', self.noun_list)
                    break

    def wordDeal(self, words):
        '''分词后的数据处理，去掉停止词以及提取其中的名词'''
        words_posseg_line = []
        noun_line_list = []
        for word, flag in words:
            # 判断是否是停止词
            if word not in self.stop_words_list:
                if flag == 'n':
                    noun_line_list.append(word)
                words_posseg_line.append((word, flag))
                self.words_posseg.append(words_posseg_line)
            else:
                continue
        self.noun_list.append(noun_line_list)
        # 保存文件
        self.saveFile(noun_line_list, 'noun_word_total')
        self.saveFile(words_posseg_line, 'seg_comment')
        print('The %d line deal with is done!' % (self.line_num))

    def saveFile(self, words_line, file_name):
        '''以字符串的形式保存文件函数'''
        with open('%s.txt' % (file_name), 'a') as f:
            f.write(str(words_line))
            f.write('\n')


class ProductFeature(object):
    """docstring for ProductFeature"""

    def __init__(self, noun_list):
        super(ProductFeature, self).__init__()
        # 名词列表
        self.noun_list = list(map(set, noun_list))
        print(self.noun_list)
        # 名词的所有候选项集的集合
        self.noun_C1_list = []
        # 满足最小支持度的频繁1项集
        self.support_L1_list = []
        # 满足最小支持度的频繁1项集以及对应的支持度字典
        self.support_data = {}
        # 最小支持度
        self.minSupport = 0.01
        # 最小置信度
        self.minConf = 0.05
        # 各阶频繁项集的列表
        self.freaquent_item_list = []
        self.big_rule_list = []
        # 直接创建最初的候选集
        self.createC1()

    def createC1(self):
        '''构建所有候选项的集合'''
        list_C1 = []
        for noun_line_list in self.noun_list:
            for word in noun_line_list:
                if [word] not in list_C1:
                    list_C1.append([word])
        list_C1.sort()
        self.noun_C1_list = map(frozenset, list_C1)
        self.support_L1_list, self.support_data = self.scanD(
            self.noun_C1_list)
        pickleDump('noun_C1_list', self.noun_C1_list)
        pickleDump('support_L1_list', self.support_L1_list)

    def scanD(self, frequent_item):
        '''扫描频繁项集并提取满足最小支持度的数据'''
        ssCnt = {}
        support_min_list = []
        support_min_data = {}
        for tid in self.noun_list:
            for can in frequent_item:
                if can.issubset(tid):
                    ssCnt[can] = ssCnt.get(can, 0) + 1
        num_noun_line = float(len(self.noun_list))
        for key in ssCnt.keys():
            support = ssCnt[key] / num_noun_line
            if support >= self.minSupport:
                support_min_list.insert(0, key)
            support_min_data[key] = support
        return support_min_list, support_min_data

    def aprioriGen(self, Lk, k):
        # 根据前一次的频繁项集和k值创建候选项集
        retList = []
        lenLk = len(Lk)
        '''
        没有使用两两之间求并集的方式来创建候选项集，因为有很多重复的，效率较低
        采用的是判断前一次的频繁项集之间前k-2项是否相同
        （其实就是除掉最后一项是否相同，因为k比前一次的频繁项集元素数多一）
        相同的话那代表最后一项绝对不同，直接求并集就可以，减少了重复计算的次数'''
        for i in range(lenLk):
            for j in range(i + 1, lenLk):
                L1 = list(Lk[i])[:k - 2]
                L2 = list(Lk[j])[:k - 2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    retList.append(Lk[i] | Lk[j])
        # 返回的就是k次的候选项集
        return retList

    def apriori(self):
        # apriori算法不断的循环寻找频繁项集，直到没有更高阶的频繁项集
        self.freaquent_item_list = [self.support_L1_list]
        k = 2
        while(len(self.freaquent_item_list[k - 2]) > 0):
            Ck = self.aprioriGen(self.freaquent_item_list[k - 2], k)
            Lk, supK = self.scanD(Ck)
            # 将每阶频繁项集中高于最低支持度的项加入到满足最小支持度的字典中
            self.support_data.update(supK)
            self.freaquent_item_list.append(Lk)
            print('%d freaquent item list is done!' % (k))
            k += 1
        pickleDump('freaquent_item_list', self.freaquent_item_list)
        pickleDump('support_data', self.support_data)

    def generateRules(self):
        for i in range(1, len(self.freaquent_item_list)):
            for freqSet in self.freaquent_item_list[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if (i > 1):
                    self.rulesFromConseq(freqSet, H1)
                else:
                    self.calcConf(freqSet, H1)

    def calcConf(self, freqSet, H):
        for conseq in H:
            conf = self.support_min_data[freqSet] / \
                self.support_min_data[freqSet - conseq]
            if conf >= self.minConf:
                print(freqSet - conseq, '-->', conseq, 'conf:', conf)
                self.big_rule_list.append((freqSet - conseq, conseq, conf))

    def rulesFromConseq(self, freqSet, H):
        m = len(H[0])
        if (len(freqSet) > (m + 1)):
            Hmp1 = self.aprioriGen(H, m + 1)
            Hmp1 = self.calcConf(freqSet, Hmp1)
            if (len(Hmp1) > 1):
                self.rulesFromConseq(freqSet, Hmp1)


def pickleDump(pickle_file_name, object_name):
    '''序列化保存变量对象'''
    with open('./pickleFile/%s.pickle' % (pickle_file_name), 'wb') as f:
        pickle.dump(object_name, f)


def pickleLoad(pickle_file_name):
    '''序列化读取变量对象'''
    with open('./pickleFile/%s.pickle' % (pickle_file_name), 'rb') as f:
        return pickle.load(f)


def main():
    jieba.load_userdict('dict.txt.big')
    phone_product_reviews = ProductReviews('phone_comment')
    phone_product_reviews.stopWordsLoad('stop_word')
    phone_product_reviews.segSentence()
    print('Comment deal with is OK!')

    product_feature = ProductFeature(phone_product_reviews.noun_list)
    product_feature.apriori()
    for i in product_feature.freaquent_item_list:
        print(str(i))


if __name__ == '__main__':
    main()
