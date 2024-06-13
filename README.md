# CS147

## Setup
First of all clone the repository. 
<br>Once you are done, install the required python packages:
```
# You can setup a virtual environment if you want
python3 -m pip install venv
python3 -m venv .venv
source .venv/bin/activate

# Install packages
python3 -m pip install -r requirements.txt
```

## Run the code
Change directory to `src` and from there you can run the targets of the makefile.
- Run `make cpu` to run the cpu version of the neural network
- Run `make gpu` to run the gpu version of the neural network
- Run `make pytorch` to run the model on pytorch

## Links
- Presentation: https://docs.google.com/presentation/d/1pMCIn6s4FMmayNxdV6SKRwrzRtKbEc6xfHKGRCjdnIM/edit?usp=sharing
- Report: https://docs.google.com/document/d/1CZPaI57Etz3DFJPeZs57x3tjNPrIG3cxlGybwlbjN1g/edit?usp=sharing
- Colab: https://colab.research.google.com/drive/1C607n23h6gnq2w5iezJEZ5IX1nj7DIPI?usp=sharing