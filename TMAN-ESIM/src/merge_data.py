import random

map = {'entailment': '0', 'contradictory': '1', 'neutral': '2'}
en = []
cn = []
with open('../../data/XNLI-MT-1.0/multinli/multinli.train.en.tsv', mode='r') as fd:
    for line in fd.readlines():
        if line.startswith('premise\thypo'):
            continue
        array = line.strip().split('\t')
        array.extend(['1'])
        en.append(array)

with open('../../data/XNLI-MT-1.0/multinli/multinli.train.zh.tsv', mode='r') as fd:
    for line in fd.readlines():
        if line.startswith('premise\thypo'):
            continue
        array = line.strip().split('\t')
        array.extend(['0'])
        cn.append(array)

en.extend(cn)
list = en
random.shuffle(list)

count = 0
fw = open('../dataset/train_enzh.txt', mode='w')
for item in list:
    fw.write(map[item[2]] + '\t' + item[0] + '\t' +item[1] + '\t' + item[3] + '\n')
    count += 1

print("Total number of merged data: " + str(count))