# -*- coding: utf-8 -*-
"""
Created by yuxi
"""

import time
from math import sin, asin, cos, sqrt, radians

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

import kmedoids
import pymysql


def haversine(lonlat1, lonlat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def clustering_by_dbscan():
    filepath = "nodes.csv"
    df = pd.read_csv(filepath)
    X = pd.DataFrame({"lat": df["lat"], "lng": df["lng"]})

    distance_matrix = squareform(pdist(X, (lambda u, v: haversine(u, v))))

    db = DBSCAN(eps=2, min_samples=5, metric='precomputed')  # using "precomputed" as recommended
    y_db = db.fit_predict(distance_matrix)

    # db = DBSCAN(eps=2 / 6371., min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coordinates))

    X['cluster'] = y_db

    results = {}
    for i in X.values:
        if i[2] not in results.keys():
            results[i[2]] = [(i[0], i[1])]
        else:
            if results[i[2]]:
                results[i[2]].append((i[0], i[1]))
            else:
                results[i[2]] = [(i[0], i[1])]
    print("\nDBSCAN实现地理位置聚类：")
    print(results.keys())
    for k in results.keys():
        print(k, ": ", results[k])

    plt.scatter(X['lat'], X['lng'], c=X['cluster'])
    # plt.show()

    time.sleep(1)


def clustering_by_kmedoids():
    data = np.array([[116.455788, 39.920767], [116.456065, 39.920965], [116.452312, 39.92304], [116.421385, 39.989539],
                     [116.455685, 39.92069], [116.455876, 39.920845], [116.455973, 39.920902], [116.455645, 39.920657],
                     [116.456022, 39.920934], [116.455685, 39.920691], [116.456023, 39.920671], [116.45596, 39.920864],
                     [116.455522, 39.920856], [116.455276, 39.920407], [116.455799, 39.920867],
                     [116.455349, 39.920425], [116.45511, 39.920377], [116.455318, 39.920442], [116.455298, 39.920474],
                     [116.455839, 39.920636], [116.455979, 39.921168], [116.454281, 39.920006], [116.45598, 39.920612],
                     [116.45388, 39.919584], [116.455474, 39.920737], [116.456009, 39.920641], [116.455439, 39.920574],
                     [116.455759, 39.920841], [116.455838, 39.920644], [116.455983, 39.920847],
                     [116.459803, 39.922041], [116.456029, 39.92088], [116.455539, 39.920603], [116.455989, 39.920851],
                     [116.455719, 39.920789], [116.45601, 39.92082], [116.456229, 39.920564], [116.455906, 39.920771],
                     [116.456248, 39.920868], [116.455805, 39.920544], [116.455896, 39.920758], [116.43692, 39.926767],
                     [116.454672, 39.92024], [116.454813, 39.917848], [116.381415, 40.00875], [116.422925, 39.980757],
                     [116.422849, 39.9808], [116.38107, 40.009217], [116.456078, 39.920747], [116.455242, 39.919515],
                     [116.455615, 39.920533], [116.422092, 39.991104], [116.454847, 39.917724],
                     [116.456686, 39.924316], [116.45575, 39.920642], [116.456713, 39.924413], [116.455846, 39.920828],
                     [116.422108, 39.991098], [116.422075, 39.991139], [118.775572, 31.97337], [118.776968, 31.97392],
                     [118.778187, 31.973121], [118.775695, 31.973254], [118.775302, 31.973807],
                     [118.776303, 31.973692], [118.777541, 31.973439], [118.776196, 31.973489],
                     [116.448944, 39.926799], [116.45487, 39.917804], [116.455762, 39.920645], [116.456146, 39.920441],
                     [116.455857, 39.920043], [116.455458, 39.920826], [116.455533, 39.920791],
                     [116.455426, 39.920896], [116.45566, 39.920811], [116.455696, 39.920621], [116.453667, 39.9259],
                     [116.466606, 39.886322], [116.455917, 39.92062]])

    # distance matrix
    D = pairwise_distances(data, metric='euclidean')

    # split into 2 clusters
    M, C = kmedoids.kMedoids(D, 4)

    print("\nK-Medoids实现地理位置聚类：")
    print('中心点:')
    for point_idx in M:
        print(data[point_idx])
    print('簇:')
    for label in C:
        for point_idx in C[label]:
            print('label {0}:　{1}'.format(label, data[point_idx]))

    time.sleep(1)


def clustering_by_dbscan_and_kmedoids():
    df = pd.read_csv("/location_clustering_kmedoids_dbscan/nodes.csv")

    X = pd.DataFrame({"lat": df["lat"], "lng": df["lng"]})

    distance_matrix = squareform(pdist(X, (lambda u, v: haversine(u, v))))

    db = DBSCAN(eps=2, min_samples=5, metric='precomputed')  # using "precomputed" as recommended
    y_db = db.fit_predict(distance_matrix)

    X['cluster'] = y_db

    results = {}
    for i in X.values:
        if i[2] not in results.keys():
            results[i[2]] = [[i[1], i[0]]]
        else:
            if results[i[2]]:
                results[i[2]].append([i[1], i[0]])
            else:
                results[i[2]] = [[i[1], i[0]]]

    print("\nDBSCAN和K-Medoids结合实现地理位置聚类：")
    print("DBSCAN聚类输出: ")
    print(len(results), results.keys())
    print("KMedoids每个簇: ")
    for k in results.keys():
        data = numpy.array(results[k])
        # distance matrix
        D = pairwise_distances(data, metric='euclidean')
        # split into 2 clusters
        M, C = kmedoids.kMedoids(D, 1)
        print('中心点:')
        for point_idx in M:
            print(data[point_idx])
        print('簇:')
        for label in C:
            for point_idx in C[label]:
                print(data[point_idx])
        print("\n")

    print("聚类结果导入数据库完成")
    print("完成网络初始化")
    time.sleep(1)


def get_data(file_name):
    # 用pandas读取csv
    # data = pd.read_csv(file_name,engine='python',encoding='gbk')
    data = pd.read_csv(file_name, engine='python')
    print(data.head(5))  # 打印前5行

    # 数据库连接
    conn = pymysql.connect(
        user="root",
        port=3306,
        passwd="root",
        db="blockchain_manager",
        host="localhost",
        charset='utf8'
    )

    # 使用cursor()方法获取操作游标
    cursor = conn.cursor()

    # 数据过滤，替换 nan 值为 None
    data = data.astype(object).where(pd.notnull(data), None)

    for id, create_time, update_time, ip, name, app_id, lat, lng, port, group_id, leader, credit_token in zip(
            data['id'], data['create_time'], data['update_time'], data['ip'],
            data['name'], data['app_id'], data['lat'], data['lng'],
            data['port'], data['group_id'], data['leader'], data['credit_token']):

        # CREATE_DATE = format_date(CREATE_DATE)

        dataList = [id, create_time, update_time, ip, name, app_id, lat, lng, port, group_id, leader, credit_token]

        print(dataList)  # 插入的值

        try:
            insertsql = "INSERT INTO `member` (id, create_time, update_time, ip, name, app_id, lat, lng, port, group_id, leader, credit_token) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(insertsql, dataList)
            conn.commit()
        except Exception as e:
            print("Exception")
            print(e)
            conn.rollback()

    cursor.close()
    # 关闭数据库连接
    conn.close()


if __name__ == "__main__":
    # clustering_by_dbscan()
    # clustering_by_kmedoids()
    clustering_by_dbscan_and_kmedoids()
    # get_data('nodes.csv')
