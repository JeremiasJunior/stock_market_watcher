import collections

dic = collections.defaultdict(list)
for i in range(10):
    dic[str(i)].append(1)

for i in range(10):
    dic[str(i)].append(2)

print(dic)