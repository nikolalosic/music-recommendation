from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse
import calculations_no_item_filtering as calcs
import ContentBasedFiltering as cbf


def main():
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
    tid_tag = pd.read_sql_query("SELECT * FROM tid_tag tt WHERE tt.tid in ("+str(",".join(map(lambda x: "\'"+str(x)+"\'", existing_tids_new_id)))+")", cnx)
    tid_tag = tid_tag.rename(index=str, columns={"tid": "new_tid_id", "tag": "new_tag_id"})
    # join tid tag sa tid_tag
    tid_tag = pd.merge(tid_tag, tags, on=['new_tag_id', 'new_tag_id'])
    tid_tag = pd.merge(tid_tag, tids, on=['new_tid_id', 'new_tid_id']) # ([tid_tag, tids, tags], ignore_index=False)
    # todo: spojiti za jedan tid tagove razdvojene razmakom - gubi se val, mozda spojiti sa istim val
    tid_tag = tid_tag.groupby('tid', as_index=False).agg(lambda x: ', '.join(map(str,x.tolist())))
    tid_tag = tid_tag.drop(['new_tag_id', 'new_tid_id'], axis=1)
    tid_tag = tid_tag.rename(index=str, columns={"tid": "track_id"})
    cnx = sqlite3.connect(artist_tag_path)
    artist_term = pd.read_sql_query("SELECT * FROM artist_term", cnx)
    # spojiti da termini budu odvojeni zarezom
    artist_term = artist_term.groupby('artist_id', as_index=False).agg(lambda x: ', '.join(map(str,x.tolist())))
    artist_mbtag = pd.read_sql_query("SELECT * FROM artist_mbtag", cnx)
    # spojiti da termini budu odvojeni zarezom
    artist_mbtag = artist_mbtag.groupby('artist_id', as_index=False).agg(lambda x: ', '.join(map(str,x.tolist())))
    # spojiti sve da bude u metadata
    metadata = pd.merge(metadata, tid_tag, on=['track_id', 'track_id'])
    metadata = pd.merge(metadata, artist_term, on=['artist_id', 'artist_id'])
    metadata = pd.merge(metadata, artist_mbtag, on=['artist_id', 'artist_id'])
    print("data loaded @", datetime.now())

    print("fit model @", datetime.now())
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

    print("model fitted @", datetime.now())

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

    print("Calculating MAP @", datetime.now())
    mapr = calcs.calcMeanAveragePrecision(listenersRecs, finalListeners)
    print("-----MAP-----")
    print(mapr)

    print("Calculating NDCG @", datetime.now())
    ndcg = calcs.calcNdcg(listenersRecs, finalListeners)
    print("-----NDCG ----")
    print(ndcg)

if __name__ == '__main__':
    main()
    print("main end ", datetime.now())