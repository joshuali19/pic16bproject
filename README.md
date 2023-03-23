# Project Title

Spotify Recommendation System for Playlists

## Description

With streaming services like Spotify, we have access to millions of songs at our fingertips. However, with so many options, it can be difficult to discover new artists and songs that we might enjoy. This is where recommendation systems come in.

This blog post is a recap of a final project for PIC16B W23 at UCLA. The goal of this project is to create a song recommendation system that would recommend songs for a given playlist. To do this, we utilized the Million Playlist Dataset, which Spotify provided for this purpose. There are 1 million playlists within this dataset, as the name suggests. With this data, we conducted our project through 4 main steps:

1. Acquiring the data necessary for recommendation
2. Cleaning the data
3. Building the Recommendation System
4. Connecting it with a website to host the recommendation system

## Getting Started

### Dependencies

This was all run on Mac using Python 3 and Jupyter Notebook.

Here are the libraries need to run the program.

```python
# web stuff
from flask import Flask, g, render_template, request

# data cleaning/opening
import pandas as pd
import sklearn as sk
import numpy as np
import sqlite3
import pandas as pd
import json
import os
import io
import base64

# spotify API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# calculating metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
```

### Installing

Download the './website' folder of our repo. You will need to modify the Spotify Credentials in `app.py	` before you run it. Go to [Spotify For Developers](https://developer.spotify.com/dashboard/) to create your own credentials

```python
os.environ["SPOTIPY_CLIENT_ID"] = "***"
os.environ["SPOTIPY_CLIENT_SECRET"] = "***"
os.environ['SPOTIPY_REDIRECT_URI'] = "***"
sp = spotipy.Spotify(client_credentials_manager =      
                     SpotifyClientCredentials(requests_timeout = 10))
```

### Executing program

To run it on your local server, open up a terminal and run these lines (assuming dependencies are installed):

```
export FLASK_ENV=development
flask run
```

Then, copy the local host link and paste it onto your browser. You should be able to get to the home page.
Import a playlist as a JSON file, or input a playlist id.
Click Submit, and watch your recommendations pop up.

## Authors

Joshua Li (joshuali19@g.ucla.edu)
Antoine Neveu (aneveu23@g.ucla.edu)
Robert Tran (robertktran@g.ucla.edu)
