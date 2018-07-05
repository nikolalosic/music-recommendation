# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse
import calculations_no_item_filtering as calcs

def appendTestData(listenerID, songs, totalPlays, listenersTotalPlays, listenersSongs, songsListeners, listenerIDs):
    """
    Called by traverseFile, puts test data into storage dictionaries.
    
    Does everything appendTrainData does, and adds listenerIDs of test listeners to set.
    
    """
    appendTrainData(listenerID, songs, totalPlays, listenersTotalPlays, listenersSongs, songsListeners)
    listenerIDs.add(listenerID)


def appendTrainData(listenerID, songs, totalPlays, listenersTotalPlays, listenersSongs, songsListeners):
    """
    Called by traverseFile, puts training data into storage dictionaries.
    
    """
    for song, playCount in songs.items():
        playPercent = playCount / float(totalPlays)
        songs[song] = playPercent

        if song not in songsListeners:
            songsListeners[song] = set(listenerID)
        else:
            songsListeners[song].add(listenerID)

    listenersTotalPlays[listenerID] = totalPlays
    listenersSongs[listenerID] = songs


def appendAnswersData(listenerID, songs, totalPlays, listenersAnswersLists):
    """
    Called by traverseFile, appends to dictionary of key = listenerID, value = songs this listener listened to (in answers data), 
    songs per listener are in no particular order
    """
    songList = list(songs.keys())  # song list for this listenerID
    listenersAnswersLists[listenerID] = songList


def traverseFile(dataFile, fcn, args):
    """
    Iterates over dataFile and executes fcn when listenerID changes.
    
    Used for populating training, test, and hidden data into dictionaries where they're needed.
    
    Train and test data populate listenersTotalPlays, listenersSongs, songsListeners.
    Test data also needs to fill listenerIDs set with the id of each listener in the test data.
    Hidden (i.e. answers) data only needs to produce a dictionary of 
    key = listenerID, value = list of songs listened to (order doesn't matter)
    
    Params (only those that seem to need comments)
    ______
        
    fcn: the function to execute
    args: addt'l arguments to fcn
    
    """
    listenerID = None
    songs = {}  # dict where keys are songID, values are % play count (of user's total) -- value of listenersSongs dict
    totalPlays = 0  # total play count for a given listener
    lineCount = 0
    with open(dataFile, 'r') as openFile:
        for line in openFile:
            thisLine = line.replace('\n', '').split('\t')
            if thisLine[0] != listenerID:
                if listenerID is not None:
                    fcn(listenerID, songs, totalPlays, *args)
                listenerID = thisLine[0]
                songs = {}
                totalPlays = 0
            lineCount += 1
            if lineCount % 1000000 == 0:
                print(lineCount)
            songs[thisLine[1]] = int(thisLine[2])
            totalPlays += int(thisLine[2])
            # if lineCount == 20000000:
            #    break

    # to add for last user in file, as will have finished 'for' loop, and exited 'with'
    fcn(listenerID, songs, totalPlays, *args)


def populateTestStorage(dataFile, listenersTotalPlays, listenersSongs, songsListeners):
    """
    Calls traverseFile with key argument (appendTestData function) and following argument (the quartet)
    
    Returns
    _______
    set of listenerIDs of listeners in test data
    """
    listenerIDs = set()
    traverseFile(dataFile, appendTestData, [listenersTotalPlays, listenersSongs, songsListeners, listenerIDs])
    return listenerIDs


def populateTrainStorage(dataFile, listenersTotalPlays, listenersSongs, songsListeners):
    """
    Calls traverseFile with key arguments (appendTrainData function) and following argument (the triplet)
    """
    traverseFile(dataFile, appendTrainData, [listenersTotalPlays, listenersSongs, songsListeners])


def populateAnswersStorage(dataFile):
    """
    Calls traverseFile with key arguments (appendAnswersData function) and following argument 
    
    Returns
    _______
    dictionary, key = listenerID, value = list (in no particular order) of songs in hidden data this listener listened to
    """
    listenersAnswersLists = {}  # see above
    traverseFile(dataFile, appendAnswersData, [listenersAnswersLists])
    return listenersAnswersLists


def loadSimilarsAndSongToTrackFromDatabase():
    train_path = 'data/10000.txt'
    track_meta_path = 'data/subset_track_metadata.db'
    track_tags_path = 'data/lastfm_tags.db'
    artist_tag_path = 'data/subset_artist_term.db'
    test_path = 'data/year1_test_triplets_visible-test.txt'
    final_test_path = 'data/year1_test_triplets_hidden-test.txt'

    print("load data @", datetime.now())
    dataset = pd.read_csv(train_path, delimiter="\t", names=["listener_id", "song_id", "listen_count"])
    testset = pd.read_csv(test_path, delimiter="\t", names=["listener_id", "song_id", "listen_count"])
    finalset = pd.read_csv(final_test_path, delimiter="\t", names=["listener_id", "song_id", "listen_count"])
    dataset_and_testset = pd.concat([dataset, testset]).reset_index(drop=True)
    cnx = sqlite3.connect(track_meta_path)
    metadata = pd.read_sql_query("SELECT * FROM songs", cnx)
    existing_tids = metadata["track_id"].values
    cnx = sqlite3.connect(track_tags_path)
    tids = pd.read_sql_query("SELECT * FROM tids", cnx)
    # dodat redni broj kolonu u tids
    tids.insert(0, 'new_tid_id', range(1, 1 + len(tids)))
    # izbaciti tid-ove ne definisane gore u existing_tids
    tids = tids.loc[tids['tid'].isin(existing_tids)].reset_index(drop=True)
    tags = pd.read_sql_query("SELECT * FROM tags", cnx)
    # dodat redni broj kolonu u tags
    tags.insert(0, 'new_tag_id', range(1, 1 + len(tags)))
    # selektovati samo tid_tag-ove sa tid-ovima iz liste gore
    existing_tids_new_id = tids["new_tid_id"].values
    tid_tag = pd.read_sql_query("SELECT * FROM tid_tag tt WHERE tt.tid in (" + str(
        ",".join(map(lambda x: "\'" + str(x) + "\'", existing_tids_new_id))) + ")", cnx)
    tid_tag = tid_tag.rename(index=str, columns={"tid": "new_tid_id", "tag": "new_tag_id"})
    # join tid tag sa tid_tag
    tid_tag = pd.merge(tid_tag, tags, on=['new_tag_id', 'new_tag_id'])
    tid_tag = pd.merge(tid_tag, tids, on=['new_tid_id', 'new_tid_id'])  # ([tid_tag, tids, tags], ignore_index=False)
    # todo: spojiti za jedan tid tagove razdvojene razmakom - gubi se val, mozda spojiti sa istim val
    tid_tag = tid_tag.groupby('tid', as_index=False).agg(lambda x: ', '.join(map(str, x.tolist())))
    tid_tag = tid_tag.drop(['new_tag_id', 'new_tid_id'], axis=1)
    tid_tag = tid_tag.rename(index=str, columns={"tid": "track_id"})
    cnx = sqlite3.connect(artist_tag_path)
    artist_term = pd.read_sql_query("SELECT * FROM artist_term", cnx)
    # spojiti da termini budu odvojeni zarezom
    artist_term = artist_term.groupby('artist_id', as_index=False).agg(lambda x: ', '.join(map(str, x.tolist())))
    artist_mbtag = pd.read_sql_query("SELECT * FROM artist_mbtag", cnx)
    # spojiti da termini budu odvojeni zarezom
    artist_mbtag = artist_mbtag.groupby('artist_id', as_index=False).agg(lambda x: ', '.join(map(str, x.tolist())))
    # spojiti sve da bude u metadata
    metadata = pd.merge(metadata, tid_tag, on=['track_id', 'track_id'])
    metadata = pd.merge(metadata, artist_term, on=['artist_id', 'artist_id'])
    metadata = pd.merge(metadata, artist_mbtag, on=['artist_id', 'artist_id'])

    con = sqlite3.connect(track_meta_path)
    df2 = pd.read_sql_query("SELECT * FROM songs", con)
    track_to_song = df2.set_index('track_id').to_dict()['song_id']
    song_to_track = df2.set_index('song_id').to_dict()['track_id']
    print("data loaded @", datetime.now())

    print("create similarity matrix @", datetime.now())
    ds = metadata.drop(['artist_hotttnesss', 'artist_familiarity',
                        'artist_mbid', 'artist_id', 'track_id'], axis=1)

    tf = TfidfVectorizer(analyzer='word',
                         ngram_range=(1, 3),
                         min_df=0,
                         stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['mbtag'])
    tfidf_matrix1 = tf.fit_transform(ds['term'])
    tfidf_matrix2 = tf.fit_transform(ds['tag'])
    # tfidf_matrix3 = tf.fit_transform(ds['artist_name'])
    tfidf_matrix4 = tf.fit_transform(ds['release'])

    combined = sparse.hstack((
        tfidf_matrix,
        tfidf_matrix1,
        tfidf_matrix2,
        # tfidf_matrix3,
        tfidf_matrix4
    ))

    cosine_similarities = linear_kernel(combined, combined)
    song_reco = {}
    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(ds['song_id'][i], cosine_similarities[idx][i])
                         for i in similar_indices]

        id = row['song_id']
        similar_items = [it for it in similar_items if it[0] != id]
        # This 'sum' is turns a list of tuples into a single tuple:
        # [(1,2), (3,4)] -> (1,2,3,4)
        # flattened = sum(similar_items, ())
        # print("Top 10 recommendations for %s: %s" % (id, flattened))
        song_reco[id] = dict(similar_items)
        # vazi i u suprotnom smjeru
        for i in song_reco[id]:
            if i not in song_reco:
                song_reco[i] = {}
            song_reco[i][id] = song_reco[id][i]

    print("created similarity matrix @", datetime.now())

    return (song_reco, track_to_song, song_to_track)

if __name__=='__main__':
    similars, songToTrack, trackToSong = loadSimilarsAndSongToTrackFromDatabase()
    print(similars)
    # listenersRecs = ContentBasedFiltering.contentBasedFiltering(songsListeners, listenersSongs, listenersRecs, similars,
    #                                                             songToTrack, trackToSong)
