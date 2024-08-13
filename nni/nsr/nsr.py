class Value:
    def __init__(self,data):
        self.data=data
    def __add__(self,other):
        other=other if isinstance(other, Value) else Value(other)
        out=Value(self.data+other.data)
        return out
a=Value(5)
b=Value(6)
c=Value(7)
d=Value(7)
r=a+b+c+d
print(r.data)