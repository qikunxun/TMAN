import random
en = []
cn = []
with open('../data/XNLI-MT-1.0/multinli/multinli.train.en.tsv', mode='r') as fd:
    for line in fd.readlines():
        if line.startswith('premise\thypo'):
            continue
        array = line.strip().split('\t')
        array.extend(['source'])
        en.append(array)

with open('../data/XNLI-MT-1.0/multinli/multinli.train.zh.tsv', mode='r') as fd:
    for line in fd.readlines():
        if line.startswith('premise\thypo'):
            continue
        array = line.strip().split('\t')
        array.extend(['target'])
        if(len(array[0]) > 0 and len(array[1]) > 0):
            cn.append(array)

en.extend(cn)
list = en
# random.shuffle(list)
count = 0
fw = open('../data/XNLI-MT-1.0/multinli/multinli.train.en-zh.tsv', mode='w')
fw.write('premise\thypo\tlabel\tlanguage\n')
for item in list:
    fw.write(item[0] + '\t' + item[1] + '\t' +item[2] + '\t' + item[3] + '\n')
    count += 1

print('done')