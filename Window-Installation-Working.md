Create conda environment in a particular directory

    conda create --prefix ./env python=3.7

Install fbprophet

    conda install -c conda-forge fbprophet

Install Kats

    pip install kats

Downgrade Packaging to 21.3

Change the line 178 in the site-packages\holidays\registry.py file in conda environment 
from super().init(*args, **kwargs) to super().init() 