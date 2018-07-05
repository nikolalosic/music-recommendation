# for every user and his list of songs apply content based filtering
def contentBasedFiltering(songsListeners, listenersSongs, listenersRecs, similars, trackToSong, songToTrack, listenersAnswersLists):
    # firstTenMap(similars)
    # firstTen(trackToSong)

    lr2 = {}
    for user in listenersAnswersLists:
        nl = {}
        for song in listenersRecs[user]:
            if song in similars:
                mn = 0
                for s in similars[song]:
                    nl[s] = similars[song][s]
                    mn += similars[song][s]
                nl[song] = mn/len(similars[song])
            else:
                nl[song] = 5
        lr2[user] = list(map(lambda x: x[0], sorted(nl.items(), key=lambda kv: kv[1], reverse=True)))


        # for key, value in listenersRecs.items():
            # print("lengths before: ", len(value))
            # listLen = len(value)
            # filter(value, listenersSongs[key], similars, trackToSong, songToTrack, len(songsListeners))
            # print("lengths after: ", len(value))
            # del value[listLen:]

        # del listenersRecs[user][:]
        # for key, value in sorted(lr.items(), key=lambda k: (k[1], k[0]), reverse=True):
        #     listenersRecs[user].append(key)

    # firstTenMap(listenersRecs)
    # return listenersRecs
    return lr2

def bla(dataset_and_testset, finalset, song_reco):
    # za dati element iz finalset odradi predikciju
    tslid = dataset_and_testset.groupby('listener_id', as_index=True).agg(lambda x: x.tolist())
    testListeners = tslid.to_dict()['song_id']
    listenersRecs = {}
    finalListeners = finalset.groupby('listener_id').agg(lambda x: x.tolist()).to_dict()['song_id']
    # in listener recs keep user from finalListeners
    for user in finalListeners:
        nl = {}
        for song in testListeners[user]:
            if song in song_reco:
                # mn = 0
                for s in song_reco[song]:
                    nl[s] = song_reco[song][s]
                    # mn += song_reco[song][s]
                # nl[song] = mn/len(song_reco[song])
            # else:
            #     nl[song] = 5
        # print(nl)
        listenersRecs[user] = list(map(lambda x: x[0], sorted(nl.items(), key=lambda kv: kv[1], reverse=True)))


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