'''
At the command line, run 
conda activate PIC16B
export FLASK_ENV=development
flask run
'''
from flask import Flask, g, render_template, request

import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
# import pickle
import sqlite3
import pandas as pd
import json
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from collections import Counter
# from tqdm import tqdm

# Spotify credentials (unique to user) using different one to see if it works
os.environ["SPOTIPY_CLIENT_ID"] = "bdf64242b8364ab5b264d3c14e8e9af6"
os.environ["SPOTIPY_CLIENT_SECRET"] = "3ed931eb80d8412292a50a10ed96e611"
os.environ['SPOTIPY_REDIRECT_URI'] = "http://localhost:5001"
sp = spotipy.Spotify(client_credentials_manager =      
                     SpotifyClientCredentials(requests_timeout = 10))

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

import io
import base64

song_df = pd.read_csv('./songs_unique.csv') # song dataframe
feats_df = pd.read_csv('./norm_song_feats.csv') # normalized features data frame

indices = pd.Series(song_df.index, index=song_df['uri']) # indices

def get_similarity_scores(df, feat_df, uri, n, model_type = cosine_similarity):
    '''
    gets the similarity scores for songs in the dataframe.
    @ inputs:
    - df (pd.DataFrame): input dataframe with audio features
    - song_title (str): title of track
    - n (int): number of recommended songs
    - model_type (df): gets the cosine similarity of big matrix
    @ outputs:
    - pandas series. of recommended songs
    '''
    # Get song indices
    index=indices[uri]
    
    # get numeric columns from song dataframe
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_df = feat_df.select_dtypes(include=numerics)
    
    # get the cosine similarity between the song, and the other songs.
    tfidf = model_type(num_df.iloc[index].values.reshape(1, -1), num_df[:].drop(index = index))[0]
    
    # Get list of scores of similarity to songs
    score=list(enumerate(tfidf))
    
    # Sort the most similar songs
    similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
    
    # return a dictionary that can be added together
    return Counter(dict(similarity_score))


def get_top_songs(playlist, song_df, feat_df, n = 10):
    '''
    Gets the top songs of a playlist
    @ inputs:
    - playlist (dict): playlist that includes track info
    - song_df (dataframe): dataframe containing more track info
    - feat_df (dataframe): dataframe of same dimensions as song_df, normalized features
    @ outputs:
    - list of the top n songs to suggest to user.
    '''
    total_score = Counter()
    # for each track
    for track in playlist['tracks']:
        # get all the similarity scores, add it up according to song index
        if track['track_uri'] in indices.keys():
            total_score += get_similarity_scores(song_df, feat_df, 
                            track['track_uri'], 5)
        # sort it by summed values of similarity
    topn_index = indices[sorted(dict(total_score), key = lambda x: x, reverse = True)[0:n]].index
    
    # return the track names that correspond to the URI's.
    return [song_df['track_name'][song_df['uri'] == uri].values[0] for uri in topn_index]

def get_playlist_track_URIs(playlist_id):
    '''
    gets the track URIs from a Spotify playlist
    @ inputs:
    - playlist_id (str)
        The unique identifier for the Spotify playlist.
    @ outputs:
    - track_uris (list of str)
        A list of the track URIs for all tracks in the playlist.
    - track_names (list of str)
        A list of the track names for all tracks in the playlist.
    '''
    # Set up authorization using the Spotify client ID and secret
    client_credentials_manager = SpotifyClientCredentials(client_id='bdf64242b8364ab5b264d3c14e8e9af6', client_secret='3ed931eb80d8412292a50a10ed96e611')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Get track URIs and names from playlist
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    playlist_tracks = []
    for track in tracks:
        track_uri = track['track']['uri']
        playlist_tracks.append({'track_uri': track_uri})
    
    print({"tracks": playlist_tracks})
    return {"tracks": playlist_tracks}

def get_file(request):
    '''
    gets the file from the request.
    '''
    
    # reads file from request
    file = request.files["playlist"]
    file_data = file.read().decode("utf-8")
    file.close()
    
    playlist_data = json.loads(file_data) # loads it through json
    return playlist_data

### stuff from last class
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def recommend():
    if request.method == 'GET':
        return render_template('recommend.html') # default recommend.html display
    else: # if someone posts
        try:
            # get playlist, and find top songs
            playlist = request.form['playlist_id']
            
            # checks to see if playlist has been submitted as ID or json file
            if not playlist:
                playlist = get_file(request)
            else:
                # get track URIs and names from playlist using Spotify API
                playlist = get_playlist_track_URIs(playlist)
            
            top_songs = get_top_songs(playlist, song_df, feats_df)
            # display the top songs
            return render_template('recommend.html', recs = top_songs)
        except:
            # return an error
            return render_template('recommend.html', error = True)

@app.route('/about/')
def about():
    try:
        # about page for creators
        return render_template('about.html', msgs = msgs)
    except:
        # return an error
        return render_template('about.html', error = True)
