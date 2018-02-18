from joblib import Parallel, delayed
import multiprocessing
import time
# what are your inputs, and what operation do you want to
# perform on each input. For example...

#inputs = range(50000000)

listenersTotalPlays = {}  # keys are listenerIDs, values are total play count for corresponding listener
listenersSongs = {}  # keys are listenerIDs, values are dictionaries of songs : play counts per each that this listener listened to
songsListeners = {}  # keys are songIDs, values are sets of listeners
listenerIDs = set()

# 450.97705149650574
def processInput(i):
    # print(i)
    return i * i


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


def processLine(line):
    listenerID = None
    songs = {}  # dict where keys are songID, values are % play count (of user's total) -- value of listenersSongs dict
    totalPlays = 0  # total play count for a given listener

    thisLine = line.replace('\n', '').split('\t')
    if thisLine[0] != listenerID:
        if listenerID is not None:
            appendTrainData(listenerID, songs, totalPlays, [listenersTotalPlays, listenersSongs, songsListeners, listenerIDs])
        listenerID = thisLine[0]
        songs = {}
        totalPlays = 0
    songs[thisLine[1]] = int(thisLine[2])
    totalPlays += int(thisLine[2])


if __name__ == '__main__':
    start_time = time.time()
    lineCount = 0
    f = open('./train_triplets.txt', 'r')

    # what are your inputs, and what operation do you want to
    # perform on each input. For example...
    content = f.readlines()
    #content = range(10)
    print(content.__len__())
    content = content[:8]
    f.close()
    num_cores = multiprocessing.cpu_count()
    print("Core number:", num_cores)
    results = Parallel(n_jobs=num_cores)(delayed(processLine)(i) for i in content)  # your code
    elapsed_time = time.time() - start_time
    print(listenersTotalPlays)
    print(elapsed_time)
