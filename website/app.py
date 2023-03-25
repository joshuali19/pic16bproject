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
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from collections import Counter
# from tqdm import tqdm

# Spotify credentials (unique to user) using different one to see if it works
os.environ["SPOTIPY_CLIENT_ID"] = "37af597b7a184dc19f140e1796cc3655"
os.environ["SPOTIPY_CLIENT_SECRET"] = "9ad98a09d833493793ff942bd3271d88"
os.environ['SPOTIPY_REDIRECT_URI'] = "http://localhost:5001"
sp = spotipy.Spotify(client_credentials_manager =      
                     SpotifyClientCredentials(requests_timeout = 10))

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

import io
import base64

song_df = pd.read_csv('./songs_unique.csv') # song dataframe
feats_df = pd.read_csv('./norm_song_feats.csv') # normalized features data frame
binary_df = pd.read_csv('./itemsessionbinary.csv')
idf_df = pd.read_csv('./idf_df.csv')
tf_df = pd.read_csv('./tf_df.csv')
tf_idf_df = pd.read_csv('./tf_idf_df.csv')

binary_df = binary_df.set_index('uri')
idf_df = idf_df.set_index('uri')
#tf_df = tf_df.set_index('pid')
tf_idf_df = tf_idf_df.set_index('uri')


indices = pd.Series(song_df.index, index=song_df['uri']) # indices

def get_song_db():
    '''
    returns the database as a sql table.
    '''
    try:
        return g.song_db
    except:
        with sqlite3.connect('./song_db.sqlite') as conn:
            g.song_db = conn
            song_df.to_sql('songs', con = conn, if_exists='replace')
            return g.song_db
            
def repetition_blocker(dict, playlist):
    #iterating through each track in user input playlist
    for track in playlist['tracks']:
        #setting total similarity score negative if present
        dict[track['track_uri']] = -1

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
    #topn_index = heapq.nlargest(n, total_score.items(), key=lambda x: x[1])
    topn_index = indices[sorted(dict(total_score), key = lambda x: x, reverse = True)[0:n]].index
    
    # return the track names that correspond to the URI's.
    return [(song_df['track_name'][song_df['uri'] == items].values[0],
            song_df['artist_name'][song_df['uri'] == items].values[0],
            '//open.spotify.com/track/' + song_df['id'][song_df['uri'] == items].values[0]) for items in topn_index]

def item_similarity(uri):
    #identifying row in binary_df corresponding to input song
    song_row = binary_df.loc[uri].values.reshape(1, -1)
    #computing similarity between input song and all other songs
    similarity_scores = cosine_similarity(binary_df, song_row)
    similarity_dict = {}
    #creating a dictionary with URI as keys, associated similarity as values
    for i, score in enumerate(similarity_scores):
        similarity_dict[str(binary_df.index[i])] = score[0]
    #returning counter version of dictionary, so that we can add it to others
    return Counter(similarity_dict)

def get_top_items(playlist, n = 10):
    #creating an empty counter to store total similarity score
    total_score = Counter()
    for track in playlist['tracks']:
        #using addition assignment to add each similarity score
            if track['track_uri'] in binary_df.index:
                total_score += item_similarity(track['track_uri'])
    #applying IDF weighting to total similarity score
    for uri in total_score.keys():
        total_score[uri] *= idf_df.loc[uri][0]
    #preventing songs in user input playlist from being suggested
    repetition_blocker(total_score, playlist)
    #using heap to find list of URIs associated with highest total similarity scores
    top_songs = [items[0] for items in heapq.nlargest(n, total_score.items(), key=lambda x: x[1])]
    print(top_songs)
    #top_songs = heapq.nlargest(n, total_score.items(), key=lambda x: x[1])
    #returning a list of tuples containing the track name, artist name, and a link to the song for the most similar songs
    return [(song_df['track_name'][song_df['uri'] == uri].values[0],
            song_df['artist_name'][song_df['uri'] == uri].values[0],
            '//open.spotify.com/track/' + song_df['id'][song_df['uri'] == uri].values[0]) for uri in top_songs]

def is_uri_in_playlist(uri, playlist):
    #iterating through each track in user input playlist
    for track in playlist['tracks']:
        #checking if desired URI matches a track's URI
        if track['track_uri'] == uri:
            #returning true if we find a match
            return True
    #returning false otherwise
    return False

def session_similarity(playlist, n = 10):
    #creating a binary encoding of the user input playlist w.r.t. dataset
    encoded_playlist = [int(is_uri_in_playlist(uri, playlist)) for uri in binary_df.index]
    #computing similarity between input playlist and all other playlists
    session_similarity = cosine_similarity(binary_df.values.T, [encoded_playlist]).flatten()
    #creating an empty counter to store total similarity score
    total_score = {}
    #iterating through each song in dataset
    for uri in binary_df.index:
        #taking dot product of playlist similarities and song's presence in each playlist, applying IDF weighting
        total_score[uri] = sum(i[0] * i[1] for i in zip(binary_df.loc[uri].values, session_similarity)) * idf_df.loc[uri][0]
    #preventing songs in user input playlist from being suggested
    repetition_blocker(total_score, playlist)
    #using heap to find n highest total similarity scores
    top_songs = [items[0] for items in heapq.nlargest(n, total_score.items(), key=lambda x: x[1])]
    #returning a list of tuples containing the track name, artist name, and a link to the song for the most similar songs
    return [(song_df['track_name'][song_df['uri'] == uri].values[0],
            song_df['artist_name'][song_df['uri'] == uri].values[0],
            '//open.spotify.com/track/' + song_df['id'][song_df['uri'] == uri].values[0]) for uri in top_songs]

def tf_idf_encoder(uri, playlist):
    #iterating through each track in user input playlist
    for track in playlist['tracks']:
        #checking if desired URI matches a track's URI
        if track['track_uri'] == uri:
            #returning idf value if match exists
            return idf_df.loc[uri][0]
    #returning 0 otherwise
    return 0

def tf_idf_similarity(playlist, n = 10):
    #creating a tf-idf vector based on the user playlist
    tf_idf_playlist = [tf_idf_encoder(uri, playlist)*(1/(len(playlist['tracks'])+50)) for uri in tf_idf_df.index]
    #computing similarity between input playlist and all other playlists w.r.t. tf-idf metric
    tf_idf_similarity = cosine_similarity(tf_idf_df.values.T, [tf_idf_playlist]).flatten()
    #creating an empty counter to store total similarity score
    total_score = {}
    for uri in tf_idf_df.index:
        #taking dot product of playlist tf_idf similarities and song's presence in each playlist - NO WEIGHTING
        total_score[uri] = sum(i[0] * i[1] for i in zip(binary_df.loc[uri].values, tf_idf_similarity))
    #preventing songs in user input playlist from being suggested
    repetition_blocker(total_score, playlist)
    #using heap to find n highest total similarity scores
    top_songs = [items[0] for items in heapq.nlargest(n, total_score.items(), key=lambda x: x[1])]
    return [(song_df['track_name'][song_df['uri'] == uri].values[0],
            song_df['artist_name'][song_df['uri'] == uri].values[0],
            '//open.spotify.com/track/' + song_df['id'][song_df['uri'] == uri].values[0]) for uri in top_songs]

def get_playlist_track_URIs(playlist_id):
    '''
    gets the track URIs from a Spotify playlist
    @ inputs:
    - playlist_id (str)
        The unique identifier for the Spotify playlist.
    @ output:
    dict containing: track_uris (list of str)
                         A list of the track URIs for all tracks in the playlist.
                     track_names (list of str)
                            A list of the track names for all tracks in the playlist.
    '''
    # Set up authorization using the Spotify client ID and secret
    client_credentials_manager = SpotifyClientCredentials(client_id='37af597b7a184dc19f140e1796cc3655', client_secret='9ad98a09d833493793ff942bd3271d88')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Get track URIs and names from playlist
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    playlist_tracks = []
    for track in tracks:
        track_uri = track['track']['uri']
        playlist_tracks.append({'track_uri': track_uri})
    
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
            #top_songs_item = get_top_items(playlist, n = 10)
            #top_songs_session = session_similarity(playlist, n=10)
            #top_songs_tf_idf = tf_idf_similarity(playlist, n=10)
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

@app.route('/tutorial/')
def tutorial():
    try:
        # tutorial page for how to find playlist id
        return render_template('tutorial.html', msgs = msgs)
    except:
        # return an error
        return render_template('tutorial.html', error = True)
