##
# Deep Neural Network

# Micro grad(dunder or magic methods)

```python
class Value:
    def __init__(self,data):
        self.data=data
    def __repr__(self):
        return f"a={self.data}"
    **def __add__(self,other):
        out=Value(self.data-other.data)**
        return out
    def __mul__(self,other):
        out=Value(self.data*other.data)
        return out
a=Value(10.0)
b=Value(3.0)
c=a.__add__(b) 
d=a.__mul__(b)
**a+b**

```

# _ python

Certainly! Here's a concise summary of why `_s` (or variables prefixed with a single underscore) is commonly used, without the actual code implementation:

1. **Signaling Internal Use**: `_s` signals that the variable is intended for internal use within a module or class, distinguishing it from the public interface.
2. **Encapsulation**: Encourages encapsulation by suggesting that `_s` should be accessed or modified through defined methods or properties, rather than directly.
3. **Preventing Name Clashes**: Reduces the likelihood of naming conflicts with external libraries or modules.
4. **Readability and Documentation**: Enhances code readability and documentation by clearly indicating the intended usage of variables.
5. **Consistency**: Aligns with industry best practices and conventions, promoting consistency across Python codebases.



CNN Done
For unsupervised learning Auto encoders are used in Convolution neural networks


## 02-optimization-and-regularization

| Concept         | Complete |
|-----------------|-------|
| weights-decay   |     |
| relu            |    |
| residuals       |       |
| dropout         |  ✅  |
| batch-norm      |       |
| layer-norm      |       |
| gelu            |    |
| adam            |       |
| early-stopping  |      |




## Creating Custom Dataset

