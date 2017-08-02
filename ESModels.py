import numpy as np
import pandas as pd
from scipy import sparse

class ESModels(object):

    def __init__(self, mean_bool = True, trend_bool = True, seasonal_bool = True):

        """
        :param mean_bool: Boolean, Additive ES with Level
        :param trend_bool: Boolean, Additive ES with Trend
        :param seasonal_bool: Bolean, Additive ES with Seasonality
        """

        # Configure Component of the Additive ES model
        self.mean = mean_bool
        self.trend = trend_bool
        if self.mean and not self.trend:
            raise ValueError('ES with Trend but without Mean is impossible')

        self.seasonal_bool = seasonal_bool
        self.number_seasonality = 0
        if self.seasonal_bool:
            self.seasonalities = {}

        # Store number of components
        self.components_list = {}
        self.number_components = 0
        self.number_components += 1 if mean_bool else self.number_components
        self.number_components += 1 if trend_bool else self.number_components

        self.number_components_tot = int(self.number_components)

        return


    def add_seasonality(self, period):

        """
        :param period: Integer, Periodicity of the seasonality
        :return: None
        """

        name = 'S' + str(self.number_seasonality + 1)

        if self.seasonal_bool:
            if name not in self.seasonalities.keys():

                self.seasonalities[name] = period
                self.number_seasonality = len(self.seasonalities.keys())
                self.number_components += 1
                self.number_components_tot += period

            else:
                print('Seasonality name already exists : ', name)

        else:
            raise ValueError('You have to activate seasonality to add a new seasonality')

        return

    def _create_gmat(self):

        """
        :return: Vector, g (See paper)
        """

        start_ind = 0
        gmat_list = []

        if self.mean:
            mean_row = np.zeros(self.number_components).tolist()
            mean_row[start_ind] = 1
            start_ind += 1
            gmat_list.append(sparse.lil_matrix([mean_row]))

        if self.trend:
            trend_row = np.zeros(self.number_components).tolist()
            trend_row[start_ind] = 1
            start_ind += 1
            gmat_list.append(sparse.lil_matrix([trend_row]))

        if self.seasonal_bool:
            for season in self.seasonalities.keys():
                seasonality_row = np.zeros(self.number_components).tolist()
                seasonality_row[start_ind] = 1
                seasonality_row_zeros = sparse.lil_matrix(
                    np.zeros((self.seasonalities[season] - 1, self.number_components)))
                gmat_list.append(sparse.lil_matrix([seasonality_row]))
                gmat_list.append(sparse.lil_matrix(seasonality_row_zeros))
                start_ind += 1

        return sparse.vstack(gmat_list)

    def _create_DF_forecast(self):

        """
        :return: pandas DataFrame, DataFrame with the forecasted value of the model
        """

        DF_forecast = {}

        if self.mean:
            DF_forecast['L'] = pd.DataFrame()

        if self.trend:
            DF_forecast['B'] = pd.DataFrame()

        if self.seasonal_bool:
            for season in self.seasonalities.keys():
                DF_forecast[season] = pd.DataFrame()

        DF_forecast['residual'] = pd.DataFrame()
        DF_forecast['Y'] = pd.DataFrame()
        DF_forecast['Yfiltered'] = pd.DataFrame()

        return DF_forecast

    def _create_A_matrix(self):

        """
        :return: Matrix, A (see Paper)
        """

        def vecmi(m, i):
            vecmi = np.zeros(m).tolist()
            vecmi[i - 1] = 1
            return vecmi

        # Get A mean / trend
        Al = None
        if self.mean and self.trend:
            Al = sparse.lil_matrix([[1, 1], [0, 1]])
        elif self.mean and not self.trend:
            Al = sparse.lil_matrix([[1]])

        # Block of seasonal patterns
        list_Am = []
        for season in sorted(self.seasonalities.keys()):
            mi = self.seasonalities[season]
            Ami = [vecmi(mi, mi)] + [vecmi(mi, i) for i in range(1, mi)]
            Ami = sparse.lil_matrix(Ami)
            list_Am.append(Ami)
        Aseasonal = sparse.block_diag(tuple(list_Am))

        # Combine Amean/trend with Aseasonal
        A = Aseasonal if Al == None else sparse.block_diag((Al, Aseasonal))

        return A

    def _create_w_vector(self):

        """
        :return: vector, w (see Paper)
        """

        w_vector = sparse.lil_matrix(np.zeros(self.number_components_tot))

        if self.mean:
            w_vector[0, 0] = 1
        if self.trend:
            w_vector[0, 1] = 1

        current_m = 0
        for season in self.seasonalities.keys():
            m = self.seasonalities[season]
            current_m += m
            w_vector[0, current_m + 1] = 1

        return w_vector

    def get_components(self):

        """
        :return: Dictionary, components of the model associated with their nick-names (L for mean, B for trend etc...)
        """

        if self.mean :
            self.components_list['mean'] = 'L'
        if self.trend :
            self.components_list['trend'] = 'B'

        for season in self.seasonalities:
            self.components_list[season] = season

        return self.components_list

    def _create_model(self):

        """
        :return: None
        """

        self.g_names = []

        if self.seasonal_bool:
            if len(self.seasonalities.keys()) == 0:
                print('Error, there should be at least one seasonal pattern if seasonal bool is activated!')

        # Map names of g components into g_names
        if self.mean:
            self.g_names.append(('L','alpha'))
            if self.trend:
                self.g_names.append(('B','beta'))

        if len(self.seasonalities) > 0:
            for ind, key in enumerate(sorted(self.seasonalities.keys())):
                self.g_names.append(('S' + str(int(ind+1)), 'gamma_' + str(int(ind + 1))))

        self.gmat = self._create_gmat()
        self.DF_forecast = self._create_DF_forecast()
        self.A_matrix = self._create_A_matrix()
        self.w_vector = self._create_w_vector()

        return