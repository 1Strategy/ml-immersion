#! /bin/env python

import pandas as pd
import numpy as np


def label_rating(row):
    if row['averageRating'] < 7.0:
        return 0
    elif row['averageRating'] >= 7.0:
        return 1
    else:
        return 'unrated'


def encode(row):
    item_list = row.genres.split(',')
    for genre in item_list:
#         print(f'Found Genre: {genre} with subset value: {row[str(genre)]}')
        row[str(genre)] = 1


def do_predict(data, endpoint):
    payload = '\n'.join(data)
    response = endpoint.predict(payload).decode('utf-8')
    return response


def batch_predict(data, batch_size):
    items = len(data)
    arrs = []
    for offset in range(0, items, batch_size):
        if offset+batch_size < items:
            results = do_predict(data[offset:(offset+batch_size)])
            arrs.extend(results)
        else:
            arrs.extend(do_predict(data[offset:items]))
        sys.stdout.write('.')
    return arrs

