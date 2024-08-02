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


### LLms

[Installing LLMs on the command line APPLICATIONS](https://parlance-labs.com/education/applications/simon_llm_cli/index.html)

[llm](https://pypi.org/project/llm/)

### Installation

Install this tool using pip:

pip install llm

Or using pipx:

pipx install llm

Or using Homebrew (see warning note):

brew install llm

### Check for the default model

llm models default

gpt-4o-mini

### Installing Googles palm llm
llm install llm-palm

llm keys set palm
Enter key: https://aistudio.google.com/app/apikey

### Changing the default  model to palm
llm models default palm

### plugin adds a model called palm

llm -m palm "hi"

### The output comes like this

(base) @NShravanReddy ➜ /workspaces/DeepLearning (main) $ llm -m palm "hi how are you?"

I am doing well, thank you for asking! How are you today?

llm logs -c to view the logs 

llm logs path

(base) @NShravanReddy ➜ /workspaces/DeepLearning (main) $ llm logs path
/home/codespace/.config/io.datasette.llm/logs.db


[pip install datasette]('https://docs.datasette.io/en/stable/installation.html')


llm install llm-cmd
It is not like cmd 

llm install llama-cpp-python

llm llama-cpp download-model \
  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin \
  --alias llama2-chat --alias l2c --llama2-chat



  llm install llm-ollama

  # Installing [Ollama](https://ollama.com/download/linux)
  curl -fsSL https://ollama.com/install.sh | sh 

Terminal 1 : ollama serve

Terminal 2L: ollama run phi3


  cat README.md | llm -s 'create code snippert for readme' >test1.md