# Notebooks

This repo contains notebooks used for the paper of RK Diagrams

## How to use

1. Install the rk_toolkit
2. Install any additional dependencies 
``` sh
python -m pip install -r requirements.txt
```
3. Run notebooks
``` sh
jupyter-notebook .
```

## Status

Currently things are being migrated over, and so may not work as intended until
the migration is cleaned. You can use the ml.kesselmanrao.com for a working version

### Known Issues

**tclError: no display name and no $DISPLAY environment variable tkinter while using Jupyter Notebooks**

If you are having an issue with running this over on a server, where display is not being set, the following will fix the display:

In the terminal, install xvfb

```
sudo apt-get install xvfb
```

In a notebook, add the following cell:

```
import os
os.system("Xvfb :1 -screen 0 720x720x16 &")
os.environ['DISPLAY'] = ":1.0"
```
