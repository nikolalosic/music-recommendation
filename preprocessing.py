# -*- coding: utf-8 -*-
import pandas
import sqlite3

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


def loadSimilarsAndSongToTrackFromDatabase(similars_path, t_metadata):
    con = sqlite3.connect(similars_path)
    con2 = sqlite3.connect(t_metadata)
    # print("start time:")
    # print(datetime.datetime.now())
    df = pandas.read_sql_query("SELECT * FROM similars_dest", con)
    # print("-------------similars------------------")
    # print(df.head())
    # print("-------------track to song-------------------------")
    df2 = pandas.read_sql_query("SELECT * FROM songs", con2)
    # print(df2.head())
    #
    # print("--------------------------------------------------")
    # df = df.join(df2.set_index('track_id'), on='tid')
    # track_to_song = df2.join(df.set_index('tid'), on='track_id')
    # print(track_to_song.head())
    similars = df.set_index('tid').to_dict()['target']
    track_to_song = df2.set_index('track_id').to_dict()['song_id']
    song_to_track = df2.set_index('song_id').to_dict()['track_id']

    con.close()
    con2.close()
    return (similars, track_to_song, song_to_track)
