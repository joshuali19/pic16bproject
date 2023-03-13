import pandas as pd

binary_df = pd.read_csv('/Users/aneveu23/Documents/GitHub/pic16bproject/itemsessionbinary.csv')
idf_df = pd.read_csv('/Users/aneveu23/Documents/GitHub/pic16bproject/idf_df.csv')
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def item_similarities(uri, binary_df):
    song_row = binary_df.loc[[uri]]
    item_sim = cosine_similarity(song_row, binary_df.drop([uri]))
    return Counter(dict(zip(binary_df.drop([uri]).index, item_sim[0])))

def session_similarities(playlist, binary_df):
    encoded_playlist = [int(track['track_uri'] in playlist['tracks']) for track in binary_df.index]
    session_sim = cosine_similarity([encoded_playlist], binary_df.T)
    return Counter(dict(zip(binary_df.columns, session_sim[0])))

def get_top_items(playlist, n = 10):
    total_score = Counter()
    for track in playlist['tracks']:
        total_score += item_similarities(track['track_uri'], binary_df)
    for uri in total_score:
        total_score[uri] *= idf_df.loc[idf_df['uri'] == uri]['idf'].values[0]
    top_items = total_score.most_common(n)
    return [item[0] for item in top_items]