import math
class Value:
    def __init__(self,data,_children=(),_op=''):
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._op=_op
        self._backward=lambda:None
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self,other):
        #here self is a and other is b
        other=other if isinstance(other, Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        
        def _backward():
            self.grad=1.0*out.grad
            other.grad=1.0*out.grad
        out._backward=_backward
        return out
    def __mul__(self,other):
        other=other if isinstance(other, Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        
        def _backward():
            self.grad=other.data*out.grad
            other.grad=self.data*out.grad
        out._backward=_backward
        return out
    def __pow__(self,other):
        assert isinstance(other, (int,float)),"only supporting int/float powers for now"
        out=Value(self.data**other,(self,),f'**{other}')

        def _backward():
            self.grad+=other*(self.data **(other-1))*out.grad
        out._backward=_backward
        return out
    def relu(self):
        out=Value(0 if self.data<0 else self.data,(self,),'ReLU')
        def _backward():
            self.graf+=(out.data>0)*out.grad
        out._backward=_backward
        return out
    def __radd__(self,other):
        return self+other
    def __rmul__(self,other):
        return self*other
    def __truediv__(self,other):
        return self*other**-1
    def __rtruediv__(self,other):
        return other*self**-1
    def __neg__(self):
        return self*-1
    def __sub__(self,other):
        return self+(-other)
    def __rsub__(self,other):
        return other+(-self)
    def tanh(self):
        n=self.data
        t=(math.exp(2*n)-1)/(math.exp(2*n)+1)
        out=Value(t,(self,),'tanh')
        def _backward():
            self.grad=(1-t**2)*out.grad
        out._backward=_backward
        return out
    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),'exp')

        def _backward():
            self.grad=out.data*out.grad
        out._backward=_backward

        return out
    def backward(self):
        o.grad=1.0
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()      
#a=Value(2.0)
#b=Value(-3.0)
#c=Value(10)
#d=a*b+c
#d
#(a.__mul__(b)).__add__(c)