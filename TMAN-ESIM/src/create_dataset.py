import os

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    make_dirs(['../dataset'])

    fw = open('../dataset/dev_zh.txt', mode='w')
    map = {'entailment': '0', 'contradiction': '1', 'neutral': '2'}
    with open('../../data/XNLI-1.0/xnli.dev.tsv', mode='r') as fd:
        for line in fd.readlines():
            if(line.startswith('language')):
                continue
            else:
                array = line.strip().split('\t')
                if(array[0] == 'zh'):
                    fw.write(map[array[1]] + '\t' + array[-3] +
                             '\t' + array[-2] + '\t' + array[-1] + '\t' + array[9] + '\n')


    fw = open('../dataset/test_zh.txt', mode='w')
    with open('../../data/XNLI-1.0/xnli.test.tsv', mode='r') as fd:
        for line in fd.readlines():
            if(line.startswith('language')):
                continue
            else:
                array = line.strip().split('\t')
                if(array[0] == 'zh'):
                    fw.write(map[array[1]] + '\t' + array[-3] +
                             '\t' + array[-2] + '\t' + array[-1] + '\t' + array[9] + '\n')
    print("done!")