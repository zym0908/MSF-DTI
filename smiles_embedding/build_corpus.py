import pandas as pd
from tqdm import tqdm
from smiles_embedding.build_vocab_util import split


def main():
    # 固定写死文件路径，按需修改这里即可
    in_path = "../data新/DrugBank新新.csv"   # 带smiles列的原始csv
    out_path = "../data新/DrugBank新新.txt"  # 输出分词语料txt

    # 读取所有SMILES
    smiles = pd.read_csv(in_path)['smiles'].values

    # 批量分词写入txt
    with open(out_path, 'w') as f:
        for sm in tqdm(smiles):
            f.write(split(sm) + '\n')
    print('Built a corpus file!')

if __name__=='__main__':
    main()


