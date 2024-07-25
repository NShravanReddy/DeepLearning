from micrograd.engine import Value

#a=Value(2)
#b=Value(3)
#c=a+b
#print(c.grad)
from micrograd import neuralnetwork

x = [Value(1),Value(1),Value(2)]
n = neuralnetwork.MLP(2,[1,2,2])

xs = [
    [Value(2.0), Value(3.0)],
    [Value(3.0), Value(-1.01)],
    [Value(-1.0), Value(0.51)],
    [Value(10.5), Value(1.0)],
    [Value(1.01), Value(1.0)],
    [Value(1.0), Value(-1.01)]
]
ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0), Value(1.0), Value(-1.0)]  # desired targets

ypred= [n(x) for x in xs]
print(ypred)
loss = sum( (yout - ygt)**2 for ygt, yout in zip(ys, ypred) )
print(loss)
loss.backward()
print(n.layers[0].neurons[0].w[0].grad)