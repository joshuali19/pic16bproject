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

song_df = pd.read_csv('./songs_1000p.csv') # initial data frame
DB_NAME = './songs_db.sqlite' # database to write to

def get_song_db():
    '''
    Retrieves the song database
    @ output:
    - g.message_db: a database storing songs
    '''
    try:
        # returns a database
        return g.songs_db
    except:
        # connect to a database
        with sqlite3.connect(DB_NAME) as conn:
            g.songs_db = conn
            
            # create a table if it doesn't exist
            cursor = conn.cursor()
            query = '''
                    CREATE TABLE IF NOT EXISTS songs (
                    danceability DOUBLE,
                    energy DOUBLE,
                    key INT,
                    loudness DOUBLE,
                    mode INT,
                    speechiness DOUBLE,
                    acousticness DOUBLE,
                    instrumentalness DOUBLE,
                    liveness DOUBLE,
                    valence DOUBLE,
                    tempo DOUBLE,
                    id TEXT PRIMARY KEY,
                    uri TEXT UNIQUE,
                    duration_ms INT,
                    time_signature INT,
                    track_name TEXT);
                    '''
            cursor.execute(query)
            # gets data from data frame, inserts it into database.
            for row in song_df.itertuples():
                cursor.execute('''
                    INSERT or REPLACE INTO songs (danceability, energy, key, loudness,
                    mode, speechiness, acousticness, instrumentalness, liveness, valence,
                    tempo, id, uri, duration_ms, time_signature, track_name)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ''',
                                (row.danceability, 
                                row.energy,
                                row.key,
                                row.loudness,
                               row.mode,
                               row.speechiness,
                               row.acousticness,
                               row.instrumentalness,
                               row.liveness,
                               row.valence,
                               row.tempo,
                               row.id,
                               row.uri,
                               row.duration_ms,
                               row.time_signature,
                               row.track_name)
                    )
                conn.commit()
            # return the database
            return g.songs_db
        
def get_song_df():
    with get_song_db() as conn:
        sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM songs
                               ''', conn)
        song_df = pd.DataFrame(sql_query, columns = ['danceability', 'energy', 'key', 'loudness',
                    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                    'tempo', 'id', 'uri', 'duration_ms', 'time_signature', 'track_name'])
        return song_df
        
def get_audio_features(song_uri):
    '''
    gets the audio features of a song
    @ inputs:
    - song_uri (str): the track's URI
    @ output:
    - gets a dataframe of the song's audio features
    '''
    song_feats = sp.audio_features(song_uri.split(':')[2])
    song_feats_df = pd.DataFrame(song_feats, range(len(song_feats)))
    cols_drop = ['track_href', 'analysis_url', 'type']
    return song_feats_df.drop(columns = cols_drop)

def insert_song(request):
    '''
    inserts song into database
    @ input:
    - song_feats_df (df): dataframe of song features
    @ output:
    None - will add observation to database.
    '''
    playlist = request.files['playlist']
    # print(playlist)
    # f = open(playlist)
    # js = f.read()
    # f.close()
    playlist_data = playlist.read()
    
    playlist.close()
    playlist_data2 = json.loads(playlist_data)
    
    with get_song_db() as conn:
        cursor = conn.cursor()
        track_uris = []
        track_names = []
        for track in playlist_data2['tracks']:
            track_uris.append(track['track_uri'])
            track_names.append(track['track_name'])
            row = get_audio_features(track['track_uri'])
            row['track_name'] = track['track_name']
            cursor.execute('''
                    INSERT or REPLACE INTO songs (danceability, energy, key, loudness,
                    mode, speechiness, acousticness, instrumentalness, liveness, valence,
                    tempo, id, uri, duration_ms, time_signature, track_name)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ''',(float(row.danceability), 
                        float(row.energy),
                        int(row.key),
                        float(row.loudness),
                       int(row['mode']),
                       float(row.speechiness),
                       float(row.acousticness),
                       float(row.instrumentalness),
                       float(row.liveness),
                       float(row.valence),
                       int(row.tempo),
                       str(row.id),
                       str(row.uri),
                       int(row.duration_ms),
                       int(row.time_signature),
                       str(row.track_name)))
            conn.commit()
        return track_uris, track_names
    

def generate_similarity_score(song_uri, track_name):
    '''
    gets the top n songs to recommend songs similar to one song.
    @ inputs:
    - song_uri (str): song's unique identifier
    @ outputs:
    - pandas series. of recommended songs
    '''
    song_feats_df = get_audio_features(song_uri)
    song_feats_df['track_name'] = track_name
    song_df = get_song_df()
        # print (df)
    index = song_df.index[0]
    nonnum_cols = ['id', 'uri', 'track_name']
    song_df_num = song_df.drop(columns = nonnum_cols)
    scaler = MinMaxScaler()
    normalized_song_df = scaler.fit_transform(song_df_num)
    # print(song_feats_df)
    tfidf = cosine_similarity(normalized_song_df[index].reshape((1, normalized_song_df.shape[1])), normalized_song_df[:])[0]
    # print(tfidf)
    
    # Get list of songs for given songs
    score=list(enumerate(tfidf))
    
    # Sort the most similar songs
    similarity_score = sorted(score,key = lambda x:x[1],reverse = True)[0:10]
    return Counter(dict(similarity_score))

# def total_scores()
### stuff from last class
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def recommend():
    if request.method == 'GET':
        return render_template('recommend.html') # default recommend.html display
    else: # if someone posts
        # try:
            # insert the message to the database
            
            track_uris, track_names = insert_song(request)
            total_scores = generate_similarity_score(track_uris[0], track_names[0])
            song_df = get_song_df()
            # display submit.html with conditions
            return render_template('recommend.html', names = track_names, total_scores = total_scores, song_df = song_df)
        # except:
            # return an error
            # return render_template('recommend.html', error = True)

@app.route('/about/')
def about():
    try:
        # get 5 random messages
        # msgs = random_messages(5)
        # display it
        return render_template('about.html', msgs = msgs)
    except:
        # return an error
        return render_template('about.html', error = True)
