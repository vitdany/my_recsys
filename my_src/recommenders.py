import os
import sys

import numpy as np
import pandas as pd
# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
# Для работы с матрицами
from scipy.sparse import csr_matrix

#module_path = os.path.abspath(os.path.join(os.pardir))
#if module_path not in sys.path:
#    sys.path.append(module_path)

from metrics import precision_at_k, recall_at_k
from utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, item_features, weighting=True):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.user_item_matrix, self.result = self.prepare_matrix(data, item_features)  # pd.DataFrame

        self.sparse_user_item = csr_matrix(self.user_item_matrix)

        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.sparse_user_item)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data: pd.DataFrame, item_features: pd.DataFrame):
        # your_code
        data.columns = [col.lower() for col in data.columns]
        data.rename(columns={'household_key': 'user_id',
                             'product_id': 'item_id'},
                    inplace=True)
        test_size_weeks = 3
        data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
        data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

        item_features.columns = [col.lower() for col in item_features.columns]
        item_features.rename(columns={'product_id': 'item_id'}, inplace=True)

        data_train = prefilter_items(data_train, item_features=item_features, take_n_popular=5000)

        user_item_matrix = pd.pivot_table(data_train,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробоват ьдругие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        result = data_test.groupby('user_id')['item_id'].unique().reset_index()
        result.columns = ['user_id', 'actual']


        return user_item_matrix, result

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(sparse_user_item, num_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=num_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(sparse_user_item.T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре



        res = [self.id_to_itemid[rec[0]] for rec in
               self.model.recommend(userid=self.userid_to_id[user],
                                    user_items=self.sparse_user_item,  # на вход user-item matrix
                                    N=N,
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[999999]],
                                    recalculate_user=True)]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        similar_users = self.model.similar_users(userid=user, N=N)
        res = [self.get_similar_items_recommendation(i[0], 1)[0] for i in similar_users]
        return res

    def get_score(self):

        self.result['als'] = self.result['user_id'].apply(lambda x: self.get_similar_items_recommendation(user= x, N=5))
        p = self.result.apply(lambda row: precision_at_k(row['als'], row['actual']), axis=1).mean()
        r = self.result.apply(lambda row: recall_at_k(row['als'], row['actual']), axis=1).mean()

        return p, r