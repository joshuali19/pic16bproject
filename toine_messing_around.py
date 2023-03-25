import pandas as pd
import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import spotipy

song_df = pd.read_csv('/Users/aneveu23/Documents/GitHub/pic16bproject/website/songs_unique.csv') # song dataframe
binary_df = pd.read_csv('/Users/aneveu23/Documents/GitHub/pic16bproject/website/itemsessionbinary.csv')
idf_df = pd.read_csv('/Users/aneveu23/Documents/GitHub/pic16bproject/website/idf_df.csv')
tf_idf_df = pd.read_csv('/Users/aneveu23/Documents/GitHub/pic16bproject/website/tf_idf_df.csv')
tf_df = pd.read_csv('/Users/aneveu23/Documents/GitHub/pic16bproject/website/tf_df.csv')
indices = pd.Series(song_df.index, index=song_df['uri']) # indices
binary_df = binary_df.set_index('uri')
idf_df = idf_df.set_index('uri')
#tf_df = tf_df.set_index('pid')
tf_idf_df = tf_idf_df.set_index('uri')

def repetition_blocker(dict, playlist):
    #iterating through each track in user input playlist
    for track in playlist['tracks']:
        #setting total similarity score negative if present
        dict[track['track_uri']] = -1

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
    #using heap to find n highest total similarity scores
    top_songs = [items[0] for items in heapq.nlargest(n, total_score.items(), key=lambda x: x[1])]
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
    print([items[1] for items in heapq.nlargest(n, total_score.items(), key=lambda x: x[1])])
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
    tf_idf_playlist = [tf_idf_encoder(uri, playlist) for uri in tf_idf_df.index]
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


#tf_df = pd.DataFrame({'tf': 1/(binary_df.sum(axis=0)+50)}, index=binary_df.columns)
#tf_df.to_csv('tf_df.csv')
#tf = np.array(tf_df['tf']).reshape((-1, 1)) 
#idf = np.array(idf_df['idf']).reshape((-1,1))
#tf_idf_prod = idf.dot(tf.T)
#print(tf_idf_prod)
#tf_idf_df = pd.DataFrame(tf_idf_prod, index=idf_df.index, columns=tf_df.index)
#print(tf_idf_df)
#tf_idf_df.to_csv('tf_idf_df.csv')
#print(binary_df)
#waffle = "spotify:track:000xQL6tZNLJzIrtIgxqSl"
#pancake = "spotify:track:006yrnQMCZpiUgkR612gC8"
#print(binary_df.loc[pancake].values)
#song_row = binary_df.loc[waffle].values
    #computing similarity between input song and all other songs
#print(song_row)
#similarity_scores = cosine_similarity(binary_df, song_row.reshape(1, -1)).flatten()
#print(similarity_scores)
#print(sum(i[0] * i[1] for i in zip(song_row, similarity_scores)))

#import spotipy
#import os
#from spotipy.oauth2 import SpotifyOAuth
#from spotipy.oauth2 import SpotifyClientCredentials
#os.environ["SPOTIPY_CLIENT_ID"] = "37af597b7a184dc19f140e1796cc3655"
#os.environ["SPOTIPY_CLIENT_SECRET"] = "9ad98a09d833493793ff942bd3271d88"
#os.environ['SPOTIPY_REDIRECT_URI'] = "http://localhost:5001"
#scope = "user-library-read"

#sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

#playlists = sp.current_user_playlists()
#print(playlists['items'][0]['tracks'])
    
    # Get track URIs and names from playlist
#def get_playlist_track_URIs(playlist_id):
   # Set up authorization using the Spotify client ID and secret
    #client_credentials_manager = SpotifyClientCredentials(client_id='37af597b7a184dc19f140e1796cc3655', client_secret='9ad98a09d833493793ff942bd3271d88')
    #sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Get track URIs and names from playlist
    #results = sp.playlist_tracks(playlist_id)
    #tracks = results['items']
    #playlist_tracks = []
    #for track in tracks:
        #track_uri = track['track']['uri']
        #playlist_tracks.append({'track_uri': track_uri})
    
    #return {"tracks": playlist_tracks}
#playlist = get_playlist_track_URIs('0puvJ1rlyRqsIQAlZUkFWW')
#top_songs = tf_idf_similarity(playlist, n = 10)
#print(top_songs)
#results = sp.playlist_tracks('0puvJ1rlyRqsIQAlZUkFWW')
#tracks = results['items']
##for track in tracks:
    #track_uri = track['track']['uri']
    #playlist_tracks.append({'track_uri': track_uri})
    
#print({"tracks": playlist_tracks})