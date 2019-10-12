f_in = open("NYSE.txt", 'r').readlines()
f_out = open("NYSE_crop.txt", 'w')

j = ''
for i in f_in:
    i_ = i.split('\t')[0].split('-')[0].split('.')[0]
    if(i_ != j):
        f_out.writelines(i_)
        f_out.writelines('\n')
    j = i_

