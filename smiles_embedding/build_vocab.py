import pickle
from collections import Counter
from build_vocab_util import WordVocab


def main():
    # ========== 固定写死所有参数，不需要终端传参 ==========
    corpus_path = "../data新/DrugBank新新.txt"  # 你的分词语料txt路径
    # ../data/BindingDB/BindingDB-train.txt
    min_freq = 1  # 最低出现频次
    vocab_size = None  # 词表最大长度，None=不限
    encoding = "utf-8"  # 文件编码
    # ====================================================

    # 打开语料文件构建词表
    with open(corpus_path, "r", encoding=encoding) as f:
        vocab = WordVocab(f, max_size=vocab_size, min_freq=min_freq)

    # 自动生成输出pkl路径
    out_path = '{}_vocab.pkl'.format(corpus_path[:-4])
    print("VOCAB SIZE:", len(vocab))
    # 保存词表
    vocab.save_vocab(out_path)


if __name__ == '__main__':
    main()
    print("运行完成")
