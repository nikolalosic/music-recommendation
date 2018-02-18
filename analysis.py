# -*- coding: utf-8 -*-

import preprocessing, ContentBasedFiltering
import calculations_no_item_filtering as calculations  # imports kaggle.py
from datetime import datetime  # for printing execution time of various stages of the program

listenersTotalPlays = {}  # keys are listenerIDs, values are total play count for corresponding listener
listenersSongs = {}  # keys are listenerIDs, values are dictionaries of songs : play counts per each that this listener listened to
songsListeners = {}  # keys are songIDs, values are sets of listeners
# later similarity calculations

print("Populating training data @", datetime.now())
#preprocessing.populateTrainStorage("./train_triplets_formatted.txt", listenersTotalPlays, listenersSongs, songsListeners)
preprocessing.populateTrainStorage("./train_triplets.txt", listenersTotalPlays, listenersSongs, songsListeners)

print("Populating test data @", datetime.now())
# returns set of TEST listeners' IDs
testListeners = preprocessing.populateTestStorage("./year1_test_triplets_visible-test.txt", listenersTotalPlays,
                                                  listenersSongs, songsListeners)

print("Making recommendations @", datetime.now())
listenersRecs = calculations.calcSimsAndRecommend(testListeners, listenersTotalPlays, listenersSongs, .3)

print("Populating answers data @", datetime.now())
listenersAnswersLists = preprocessing.populateAnswersStorage("./year1_test_triplets_hidden-test.txt")

print("Calculating MAP before adjusting @", datetime.now())
map = calculations.calcMeanAveragePrecision(listenersRecs, listenersAnswersLists)
print("-----MAP-----")
print(map)

print("Calculating NDCG before adjusting @", datetime.now())
ndcg = calculations.calcNdcg(listenersRecs, listenersAnswersLists)
print("-----NDCG ----")
print(ndcg)

print ("Adjusting recomendations @", datetime.now())

similars, songToTrack, trackToSong = preprocessing.loadSimilarsAndSongToTrackFromDatabase("./lastfm_similars.db", "./track_metadata.db")
listenersRecs = ContentBasedFiltering.contentBasedFiltering(songsListeners, listenersSongs, listenersRecs, similars, songToTrack, trackToSong)


print("Calculating MAP @", datetime.now())
map = calculations.calcMeanAveragePrecision(listenersRecs, listenersAnswersLists)
print("-----MAP-----")
print(map)

print("Calculating NDCG @", datetime.now())
ndcg = calculations.calcNdcg(listenersRecs, listenersAnswersLists)
print("-----NDCG ----")
print(ndcg)
