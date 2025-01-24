import math

def softmax(a):
    b=[math.exp(i) for i in a]
    c=0
    for j in a:
        c+=math.exp(j)
    return [k/c for k in b]
ans=softmax([1,2,3])
print(ans)