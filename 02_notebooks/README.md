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

## Running with Docker

You can find installation instructions for installing docker [here](https://docs.docker.com/desktop/)

Once docker is installed, you can run with docker via the following commands:

```
docker run -p 8080:8080 -v -v "${PWD}:/workspace" --env AUTHENTICATE_VIA_JUPYTER="mytoken" --shm-size 512m andorsk/rk_toolkit:latest
```
or docker-compose

```
docker-compose up -d 
```

Once the server is up, you can go to the URL ( often for you http://localhost:8080 ) to view the notebooks in your browser.

More information about the docker build process is [here](https://github.com/andorsk/rk_toolkit/blob/master/01_rk_workbench)

## Status

Currently in alpha. On Release: `0.1`

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

## Contributions and Licensing

[Creative Commons Licensing](https://creativecommons.org/licenses/by-nc/4.0/)
