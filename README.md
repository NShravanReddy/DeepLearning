
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




## Docker ollama
docker pull ollama/ollama

docker run -it \
    --rm \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama

   docker exec -it ollama bash

   apt-get install curl
https://github.com/NShravanReddy/DeepLearning?tab=readme-ov-file#installation
  



## Perplexica 
    mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

curl -fsSL https://ollama.com/install.sh | sh


ollama start
ollama pull llama3.1
ollama run llama3.1

ollama pull nomic-embed-text

git clone https://github.com/ItzCrazyKns/Perplexica.git

cd Perplexica

mv sample.config.toml config.toml (rename)

docker compose up -d

now a window pop up to open port 4000
docker stop $(docker ps -q)

## [Using Inspect ai]('https://inspect.ai-safety-institute.org.uk/')

pip install inspect-ai
 pip install google-generativeai

 export GOOGLE_API_KEY= [getkeyfrom]('https://aistudio.google.com/app/apikey')

 nano theory_of_mind.py


 from inspect_ai import Task, eval, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (               
  chain_of_thought, generate, self_critique   
)                                             

@task
def theory_of_mind():
    return Task(
        dataset=example_dataset("theory_of_mind"),
        plan=[
          chain_of_thought(),
          generate(),
          self_critique()
        ],
        scorer=model_graded_fact()
    )



    inspect eval theory_of_mind.py --model google/gemini-1.0-pro

    Damm success. but it took more than 25 minutes

![processing](/Screenshot%202024-08-08%20at%204.34.53 PM.png)