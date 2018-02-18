# for every user and his list of songs apply content based filtering
def contentBasedFiltering(songsListeners, listenersSongs, listenersRecs, similars, trackToSong, songToTrack):
    # firstTen(similars)
    # firstTen(trackToSong)
    # print(len(songsListeners))
    # key is userId, value is list of songs predicted for that user
    for key, value in listenersRecs.items():
        # print("lengths before: ", len(value))
        listLen = len(value)
        filter(value, listenersSongs[key], similars, trackToSong, songToTrack, len(songsListeners))
        # print("lengths after: ", len(value))
        del value[listLen:]


    return listenersRecs


# create new list of songs with userProfile (sorted songs that user likes) and return all
# that he didn't listen and are similar or are in userProfile
def filter(userProfile, listenersSongs, similars, trackToSong, songToTrack, totalSongsLenn):
    # map that contains recomended songs for user as key with score as value
    filterSongs = {}

    totalSongsLen = 600000

    i = 1
    # print("trackToSong length: ", len(trackToSong))
    for song in userProfile:
        if song not in listenersSongs:
            add_song_to_map(song, filterSongs, (totalSongsLen - i) / float(totalSongsLen))
        if song in songToTrack and songToTrack[song] in similars:
            sims = similars[songToTrack[song]]
            sims = sims.split(',')
            listOdd = sims[1::2]
            listEven = sims[::2]
            sims = dict(zip(listEven, listOdd))
            # key is similar song, value is decimal 0 -> 1 how similar is it
            for key, value in sims.items():
                if key in trackToSong and trackToSong[key] not in listenersSongs:
                    add_song_to_map(trackToSong[key], filterSongs, (totalSongsLen - i/float(value)) / float(totalSongsLen))
        i += 1

    del userProfile[:]
    for key, value in sorted(filterSongs.items(), key=lambda k: (k[1], k[0]), reverse=True):
        userProfile.append(key)


# adds song to map and sets its score to value
def add_song_to_map(song, map, value):
    if song in map:
        map[song] = (map[song] + float(value))/2.0
    else:
        map[song] = float(value)

def firstTenList(map):
    i = 0
    for key in map:
        print(str(key))
        i = i + 1
        if i == 10:
            break

def firstTenMap(map):
    i = 0
    for key, value in map.items():
        print(str(key) + ' ' + str(value))
        i = i + 1
        if i == 10:
            break
