from cvxpy import *
from ESModels import *
from scipy import sparse
import matplotlib.pyplot as plt


def window_func(L, R, x):

    """
    :param L: Float, Left window size
    :param R: Float, Right window size
    :param x: Float, Get window weight at x
    :return: Float, values of weighted window at x
    """

    if L < 0 or R < 0:
        raise ValueError('Window dimension should be positive')

    if x >= 0:
        y = (-1. / R) * x + 1
        return y if y >= 0 else 0

    elif x < 0:
        y = (1. / L) * x + 1
        return y if y >= 0 else 0

class RobustES(object):

    def _load_regularizer_init(self):

        """
        :return: None
        """

        # Store regularizer
        self.regularizer_dict = {}
        regul_ind = 0
        if self.ESMod.mean:
            self.regularizer_dict['mean'] = (regul_ind, regul_ind)
            regul_ind += 1
            if self.ESMod.trend:
                self.regularizer_dict['trend'] = (regul_ind, regul_ind)
                regul_ind += 1

        for season in sorted(self.ESMod.seasonalities.keys()):
            season_value = self.ESMod.seasonalities[season]
            self.regularizer_dict[season] = (regul_ind, regul_ind + season_value)
            regul_ind += season_value

        self.x_take_index = [tuple_dict[0] for tuple_dict in sorted(list(self.regularizer_dict.itervalues()))]

        return

    def _load_ESmatrix_init(self):

        """
        :return: None
        """

        # Store main matrices to execute the algorithm
        self.D_matrix = self._set_D_matrix()
        self.A_matrix = self.ESMod.A_matrix
        shapeA = self.A_matrix.shape
        if shapeA[0] != shapeA[1]:
            raise ValueError('A should be a square matrix!')

        self.w_vector = self.ESMod.w_vector
        self.H = self.ESMod.gmat

        # Store complete set of matrices needed to compute algorithm
        self.dim_A = shapeA[0]
        self.A_dict = { 0 : sparse.identity(self.dim_A) }
        self.a_dict = {0 : self.w_vector }

        for i in range(len(self.Y)):
            self.A_dict[i + 1] = self.A_dict[i] * self.A_matrix
            self.a_dict[i + 1] = self.w_vector * self.A_dict[i + 1]

        self.A_BIG = sparse.vstack(list(self.a_dict.itervalues())[:-1])
        self.A_BIG = self.A_BIG.tocsc()

        return

    def _load_ESModel_init(self, ESMod):

        """
        :param ESMod: Object, Additive ES Model
        :return: None
        """

        # Store ESModel
        self.ESMod = ESMod
        self.ESMod._create_model()
            # Store model parameters
        self.components_list = self.ESMod.get_components()
        self.nb_components = len(self.components_list.keys())

        return

    def _load_Yts_init(self, Y):

        """
        :param Y: Numpy array, Time series under study
        :return: None
        """

        L, R = self.window_dict['L'], self.window_dict['R']

        self.Y_old = Y
        self.N_old = len(self.Y_old)

        # Add invisible window
        self.Y = np.append(np.NaN * np.zeros(L), Y)
        self.Y = np.append(self.Y, np.NaN * np.zeros(R))
        self.N = len(self.Y)

        return

    def _load_window_init(self, window_dict):

        """
        :param window_dict: Dictionary, Window characteristics
        :return: None
        """

        # Store Window of ES cell
        self.window_dict = window_dict
        self.eta_matrix = [window_func(self.window_dict['L'] , self.window_dict['R'], i)
                           for i in range(- self.window_dict['L'], self.window_dict['R'] + 1)]

        return

    def _load_forecasting_init(self):

        """
        :return: None
        """

        # Loading DataFrame to store forecast and simulation index
        self.simulations_index = {}
        self.DF_forecast = self.ESMod._create_DF_forecast()
        self.conf_int = []

        return


    def __init__(self , Y, ESMod, window_dict):

        """
        :param Y: Numpy Array, original time series to be fitted and forecasted
        :param ESMod: ESModels Object, containing the Additive ES model that will fit the time series
        :param window_dict: Dictionnary, containing the single ES Cell properties.
        """

        # Load window
        self._load_window_init(window_dict)

        # Load time series
        self.N = len(Y)
        self._load_Yts_init(Y)

        # Load ES Model
        self._load_ESModel_init(ESMod)
        # Load ES matrices
        self._load_ESmatrix_init()
        self._load_regularizer_init()

        # Load forecasting tools
        self._load_forecasting_init()


        return

    def _set_D_matrix(self):

        """
        :return: D matrix for algorithm, taking into account possible missing values
        """

        D_matrix = np.ones(self.N)
        index_missing = pd.isnull(pd.Series(self.Y))
        index_missing = index_missing[index_missing == True].index

        # Change NaN value into 0 to prevent optimizer from failing. 0*np.NaN = np.NaN
        self.Y = pd.Series(self.Y).fillna(0).values

        for missing in index_missing:
            D_matrix[missing] = 0

        return D_matrix

    def _load_seasonalities_m(self):

        """
        :return: Integer, List : Total seasonality components, List of individual seasonality components
        """

        m = 0
        m_list = []

        for season in sorted(self.ESMod.seasonalities.keys()):

            season_value = self.ESMod.seasonalities[season]
            m_list.append(season_value)
            m += season_value

        return m, m_list

    def _update_opt_func(self, t):

        """
        :param t: Float, Time to apply the single ES cell model
        :return: None
        """

        L, R = self.window_dict['L'], self.window_dict['R']

        Y_VEC = np.matrix(self.Y[0 + t: L + R + t + 1]).T
        D_ETA_VEC = self.D_matrix[0 + t: L + R + t + 1] * self.eta_matrix
        D_ETA_VEC = sparse.diags(D_ETA_VEC)
        A_VEC = self.A_BIG[0: L + R + 1, :]

        self.opt_func += pnorm(D_ETA_VEC * (Y_VEC - A_VEC * self.xt_list[t]), 1)

        return

    def _update_opt_regularizer(self, t, lambda_noise):

        """
        :param t: Float, Time to apply the single ES cell model
        :param lambda_noise: Float, regularization factor (Denoising ES cell)
        :return: None
        """

        # Regularization R1
        for season in sorted(self.ESMod.seasonalities.keys()):
            season_reg_ind = self.regularizer_dict[season]
            self.regularizer_func += lambda_noise * tv(self.xt_list[t][season_reg_ind[0]: season_reg_ind[1]])

        return


    def _update_opt_anchor(self, t, lambda_anchor):

        """
        :param t: Time to link two consecutive ES cells
        :param lambda_anchor: Float, regularization factor (Linking the ES cells)
        :return: None
        """

        # Create anchor
        L, R = self.window_dict['L'], self.window_dict['R']
        self.anchor_func += float(lambda_anchor) * pnorm(self.A_dict[L + 1] * self.xt_list[t] - self.A_dict[L] * self.xt_list[t + 1], 1)

        return

    def _load_x_g(self):

        """
        :return: None
        """

        # DF to store output of model : x, s and gt
        self.x_names, self.g_names = zip(*self.ESMod.g_names)
        self.x_DF = pd.DataFrame(columns = list(self.x_names))
        self.g_DF = pd.DataFrame(columns = list(self.g_names))
        self.geps_DF = pd.DataFrame(columns = list(self.g_names))

        return

    def _build_optimization(self, lambda_anchor, lambda_noise):

        """
        :param lambda_anchor: Float, regularization factor (Linking the ES cells)
        :param lambda_noise: Float, regularization factor (Denoising)
        :return: cvxpy object, Convex objective function to optimize
        """

        m, _ = self._load_seasonalities_m()
        self.xt_list = [Variable(m + 2, 1) for _ in range(self.N_old)]

        # Building Single ES Cells
        self.opt_func = 0
        for i in range(self.N_old) :
            self._update_opt_func(t = i)

        # Building Dynamic ES Cells model
        self.anchor_func = 0
        for i in range(self.N_old - 1):
            self._update_opt_anchor(t = i, lambda_anchor = lambda_anchor)

        #Building Regularizers
        self.regularizer_func = 0
        for i in range(self.N_old):
            self._update_opt_regularizer(t = i, lambda_noise = lambda_noise)

        print('Building Objective')
        objective = self.opt_func + self.anchor_func + self.regularizer_func

        return objective

    def _solve_optimization(self, lambda_noise, lambda_anchor, status_opt = False):

        """
        :param lambda_noise: Float, regularization factor (Denoising)
        :param lambda_anchor: Float, regularization factor (Linking the ES cells)
        :param status_opt: Boolean, Enable to print optimization details
        :return: None
        """

        objective = self._build_optimization(lambda_noise = lambda_noise, lambda_anchor = lambda_anchor)

        # Optimization problem
        print('Optimizing')
        objective = Minimize(objective)
        prob = Problem(objective)
        prob.solve(solver = ECOS)

        if status_opt:
            print('Point in time: Initialization')
            print "status:", prob.status
            print "optimal value:", prob.value

        return

    def _collecting_results(self):

        """
        :return: None
        """

        L, R = self.window_dict['L'], self.window_dict['R']

        print('Collecting Results')

        #Saving x_DF
        # Add component of x
        for ind, x in enumerate(self.xt_list):
            # Need to multiply by A^L to get the estimate that we are interested in
            estimate_x = self.A_dict[L] * x.value
            values_key = np.array(estimate_x.T)[0]
            self.x_DF.loc[ind] = np.take(values_key, self.x_take_index)

        # Add residual
        residual = self.Y_old - self.x_DF.sum(axis=1)
        self.x_DF['Yfilter'] = self.x_DF.sum(axis=1)
        self.x_DF['Y'] = self.Y_old
        self.x_DF['residual'] = residual

        #Saving geps_DF
        # Add component of g
        self.geps_DF.loc[0] = np.zeros(self.nb_components)
        self.g_DF.loc[0] = np.zeros(self.nb_components)

        for ind, x in enumerate(self.xt_list):

            if ind > 0:
                geps = self.A_dict[L] * self.xt_list[ind].value - self.A_dict[L + 1] * self.xt_list[ind - 1].value
                self.geps_DF.loc[ind] = np.array(geps.T)[0][: self.nb_components]
                self.g_DF.loc[ind] = self.geps_DF.loc[ind] / self.x_DF.loc[ind]['residual']

        self.geps_DF['residual'] = self.x_DF['residual'].values
        self.g_DF['residual'] = self.x_DF['residual'].values

        return

    def fit(self, lambda_anchor, lambda_noise, status_opt = False):

        """
        :param lambda_noise: Float, regularization factor (Denoising)
        :param lambda_anchor: Float, regularization factor (Linking the ES cells)
        :param status_opt: Boolean, Detail about optimization
        :return: None
        """

        self._load_x_g()
        self._solve_optimization(lambda_anchor = lambda_anchor, lambda_noise = lambda_noise, status_opt = status_opt)
        self._collecting_results()

        return

    def _dict_simulation_i(self):

        """
        :return: Dictionary, containing component list for simulation i
        """

        dict_i = {'Yfiltered' : [], 'Y' : [], 'residual' : []}

        if self.ESMod.mean:
            dict_i['L'] = []
            if self.ESMod.trend:
                dict_i['B'] = []

        for s in self.ESMod.seasonalities.keys():
            dict_i[s] = []

        return dict_i

    def simulation_predict(self, forecast = 100, nb_simulations = 1, conf_int = [80, 90, 95]):

        """
        :param forecast: Integer, number of points in the future to forecast under uncertainty
        :param nb_simulations: Integer, number of simulation (path)
        :param conf_int: Float, confidence interval
        :return: None
        """

        L, R = self.window_dict['L'], self.window_dict['R']

        self.conf_int = conf_int
        self.DF_forecast = self.ESMod._create_DF_forecast()

        start = len(self.Y)
        for j in range(start + 1, start + forecast + 1):
            self.simulations_index[j] = self._dict_simulation_i()

        # Monte - carlo simulation (path simulation for forecasting under uncertainty)
        for i in range(nb_simulations):

            index_geps = self.geps_DF.index.tolist()
            index_simulation_forecast = np.random.choice(index_geps, size = forecast, replace = True)

            print('Simulation:', i, 'out of', nb_simulations)
            x_previous_var = self.A_dict[L]*self.xt_list[-1]
            x_previous = x_previous_var.value

            for forecast_ind, index_sample in enumerate(index_simulation_forecast):

                g_forecast_uncertainty = self.geps_DF.drop('residual', axis = 1).loc[index_sample].values
                residual = self.geps_DF['residual'].loc[index_sample]

                geps_sampled = self.H * np.matrix(g_forecast_uncertainty).T
                x_next = self.A_matrix * x_previous + geps_sampled

                start_ind = 0
                y_filtered = 0

                if self.ESMod.mean:
                    self.simulations_index[start + forecast_ind + 1]['L'].append(x_next[start_ind, 0])
                    y_filtered += x_next[start_ind, 0]

                    # Store B
                    if self.ESMod.trend:
                        start_ind += 1
                        self.simulations_index[start + forecast_ind + 1]['B'].append(x_next[start_ind, 0])
                        y_filtered += x_next[start_ind, 0]

                for season in sorted(self.ESMod.seasonalities.keys()):
                    season_value = self.ESMod.seasonalities[season]
                    start_ind += season_value
                    self.simulations_index[start + forecast_ind + 1][season].append(x_next[start_ind, 0])
                    y_filtered += x_next[start_ind, 0]

                self.simulations_index[start + forecast_ind + 1]['Yfiltered'].append(y_filtered)
                self.simulations_index[start + forecast_ind + 1]['Y'].append(self.simulations_index[start + forecast_ind + 1]['Yfiltered'][-1] + residual)
                self.simulations_index[start + forecast_ind + 1]['residual'].append(residual)

                x_previous = x_next

        # confidence intervals
        for ts_component in self.DF_forecast.keys():

            mean_l = []
            std_l = []

            minus_conf_l = {conf : [] for conf in conf_int }
            plus_conf_l = {conf : [] for conf in conf_int }

            for i in sorted(self.simulations_index.keys()):

                ts_sim = np.array(self.simulations_index[i][ts_component])
                mean_l.append(np.mean(ts_sim))
                std_l.append(np.std(ts_sim))

                for conf in conf_int:
                    minus_conf_l[conf].append(np.percentile(ts_sim, 0.5 * (100 - conf)))
                    plus_conf_l[conf].append(np.percentile(ts_sim, 100 - 0.5 * (100 - conf)))


            # Store statistics in DF_forecast dictionnary
            if ts_component == 'Y':
                self.DF_forecast[ts_component][ts_component + '_filter'] =\
                np.append(self.Y_old, np.NaN * np.zeros(forecast))

            elif ts_component == 'Yfiltered':
                self.DF_forecast[ts_component][ts_component + '_filter'] =\
                np.append(self.Y_old, np.NaN * np.zeros(forecast))

            else:
                self.DF_forecast[ts_component][ts_component + '_filter'] =\
                np.append(self.x_DF[ts_component].values, np.NaN * np.zeros(forecast))


            self.DF_forecast[ts_component][ts_component + '_mean'] = \
            np.append(np.NaN * np.zeros(self.N_old), mean_l)

            self.DF_forecast[ts_component][ts_component + '_std'] =\
            np.append(np.NaN * np.zeros(self.N_old), std_l)

            for conf in conf_int:

                self.DF_forecast[ts_component][ts_component + '_' + str(conf) + 'up'] = \
                np.append(np.NaN * np.zeros(self.N_old), plus_conf_l[conf])

                self.DF_forecast[ts_component][ts_component + '_' + str(conf) + 'down'] = \
                np.append(np.NaN * np.zeros(self.N_old), minus_conf_l[conf])

        return

    @staticmethod
    def _plot_legend_save(fig, ts_component, title, lgd_list, xlim, ylim, save):

        """
        :param fig: matplotlib object, figure to plot
        :param ts_component: String, ts component to plot
        :param title: String, title for plot
        :param lgd_list: List, legend for plot
        :param xlim: List of two integers, xlim for plot
        :param ylim: List of two integers, ylim for plot
        :param save:  Boolean,saving plot
        :return: matplotlib plot
        """

        # Defining fontsize
        fontsize_axis_tick = 12
        fontsize_axis_label = 18

        # Plot and save plot
        plt.title(ts_component)
        plt.xlabel('Time', fontsize = fontsize_axis_label)
        plt.ylabel(ts_component, fontsize = fontsize_axis_label)

        if len(ylim) > 0:
            plt.ylim(ylim[0], ylim[1])

        if len(xlim) > 0:
            plt.xlim(xlim[0], xlim[1])

        plt.tick_params(axis = 'x', labelsize = fontsize_axis_tick)
        plt.tick_params(axis = 'y', labelsize = fontsize_axis_tick)
        lgd = plt.legend(handles = lgd_list, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.grid(True)

        # Save if necessary
        if save:
            fig.savefig('DataExperiment/PlotPaper/' + title + '_' + ts_component + '.pdf', dpi = 300, format = 'pdf',
                        bbox_extra_artists = (lgd,), bbox_inches = 'tight')

        return

    def outlier_plot(self, percentiles = (5,95), save = False,  xlim = [], ylim = []):

        """
        :param percentiles: Tuple of two float, tail of the distribution for outlier detection
        :param save: Boolean, saving plot
        :param xlim: List of two integers, xlim for outlier plot
        :param ylim: List of two integers, ylim for outlier plot
        :return: outlier plot
        """

        if self.x_DF['residual'].empty:
            raise ValueError('One must run algorithm before being able to to get outlier plot')

        Y_series = pd.Series(self.Y_old)
        
        # Extract outlier by analysing the tail of the residual distribution
        residual_series = self.x_DF['residual']
        low_percentile = np.percentile(residual_series.values, percentiles[0])
        high_percentile = np.percentile(residual_series.values, percentiles[1])
        extract_indices = residual_series.loc[(residual_series <= low_percentile) | (residual_series >= high_percentile)].index

        # Plot outlier with Red dot
        fig = plt.figure()
        ts_plot , = plt.plot(Y_series.index, Y_series.values, label = 'Time Series')
        outlier_plot , = plt.plot(Y_series.loc[extract_indices].index, Y_series.loc[extract_indices].values, 'ro', label = 'Outliers')
        lgd_list = [ts_plot, outlier_plot]
        plt.xlim(0, self.N_old)

        RobustES._plot_legend_save(fig = fig, ts_component = 'Y', title = 'outliers', lgd_list = lgd_list, xlim = xlim, ylim = ylim, save = save)

    def _plot_component_w_uncertainty(self, xlim, ylim, title, ts_component, conf_int, save):

        """
        :param xlim: List of two integers, xlim for ts component plot
        :param ylim: List of two integers, ylim for ts component plot
        :param title: String, title for plot
        :param ts_component: String, ts component to plot
        :param conf_int: Float, confidence interval
        :param save: Boolean, saving plot
        :return: plot ts component under uncertainty
        """

        # Gather data
        main_y_filter = self.DF_forecast[ts_component][ts_component + '_filter'].values
        main_y_forecast = self.DF_forecast[ts_component][ts_component + '_mean'].values
        up_y_forecast = self.DF_forecast[ts_component][ts_component + '_' + str(conf_int) + 'up'].values
        down_y_forecast = self.DF_forecast[ts_component][ts_component + '_' + str(conf_int) + 'down'].values

        # Set up figure
        fig = plt.figure()

        filtery, = plt.plot(main_y_filter, label = 'Filter ' + ts_component)
        forecast, = plt.plot(main_y_forecast, label = 'Forecast ' + ts_component)
        up_forecast, = plt.plot(up_y_forecast, label = 'Up forecast ' + ts_component + ' ' + str(conf_int) + ' %', alpha = 0 )
        down_forecast, = plt.plot(down_y_forecast, label = 'Down forecast ' + ts_component + ' ' + str(conf_int) + ' %', alpha = 0 )

        plt.fill_between(np.arange(len(up_y_forecast)), down_y_forecast, up_y_forecast, alpha = 0.2,
                         edgecolor = '#1B2ACC', facecolor = '#FF0814', linewidth = 4, linestyle = 'dashdot',
                         antialiased = True)

        lgd_list = [filtery, forecast, up_forecast, down_forecast]

        RobustES._plot_legend_save(fig = fig, ts_component = ts_component, title = title,
                              lgd_list = lgd_list, xlim = xlim, ylim = ylim, save = save)

    def _plot_Y_w_uncertainty(self, xlim, ylim, title, conf_int, save):

        """
        :param xlim:  List of two integers, xlim for entire ts plot
        :param ylim:  List of two integers, ylim for entire ts plot
        :param title: String, title for plot
        :param conf_int: Float, confidence interval
        :param save: Boolean, saving plot
        :return: plot Y under uncertainty
        """

        ts_component_eps = 'Y'
        ts_component_x = 'Yfiltered'

        main_y_filter = self.DF_forecast[ts_component_x][ts_component_x + '_filter'].values
        main_y_forecast = self.DF_forecast[ts_component_x][ts_component_x + '_mean'].values
        up_y_forecast_x = self.DF_forecast[ts_component_x][ts_component_x + '_' + str(conf_int) + 'up'].values
        down_y_forecast_x = self.DF_forecast[ts_component_x][ts_component_x + '_' + str(conf_int) + 'down'].values
        up_y_forecast_eps = self.DF_forecast[ts_component_eps][ts_component_eps + '_' + str(conf_int) + 'up'].values
        down_y_forecast_eps = self.DF_forecast[ts_component_eps][ts_component_eps + '_' + str(conf_int) + 'down'].values

        fig = plt.figure()
        filtery, = plt.plot(main_y_filter, label = 'Filter ' + ts_component_x)
        forecast, = plt.plot(main_y_forecast, label = 'Forecast ' + ts_component_x)

        up_forecast_x, = plt.plot(up_y_forecast_x, label = 'Up forecast ' + ts_component_x + ' ' + str(conf_int) + ' %', alpha = 0 )
        down_forecast_x, = plt.plot(down_y_forecast_x, label = 'Down forecast ' + ts_component_x + ' ' + str(conf_int) + ' %', alpha = 0 )

        up_forecast_eps, = plt.plot(up_y_forecast_eps, label = 'Up forecast ' + ts_component_eps + ' ' + str(conf_int) + ' %', alpha = 0 )
        down_forecast_eps, = plt.plot(down_y_forecast_eps, label = 'Down forecast ' + ts_component_eps + ' ' + str(conf_int) + ' %', alpha = 0 )

        plt.fill_between(np.arange(len(up_y_forecast_x)), down_y_forecast_x, up_y_forecast_x, alpha = 0.9,
                         edgecolor = '#1B2ACC', facecolor = '#FF0814', linewidth = 4, linestyle = 'dashdot',
                         antialiased = True)
        plt.fill_between(np.arange(len(up_y_forecast_eps)), down_y_forecast_eps, up_y_forecast_eps, alpha = 0.2,
                         edgecolor = '#1B2ACC', facecolor = '#089FFF', linewidth = 4, linestyle = 'dashdot',
                         antialiased = True)

        lgd_list = [filtery, forecast, up_forecast_x, down_forecast_x, up_forecast_eps, down_forecast_eps]

        RobustES._plot_legend_save(fig = fig, ts_component = ts_component_eps, title = title, lgd_list = lgd_list, xlim = xlim, ylim = ylim, save = save)

    def simulation_plot(self, title, xlim = [], ylim = [], conf_int = None, save = False):

        """
        :param title: String, title for plot
        :param xlim: List of two integers, xlim for entire ts forecasting plot
        :param ylim: List of two integers, ylim for entire ts forecasting plot
        :param conf_int: Float, confidence interval
        :param save: Boolean, saving plot
        :return: plot Y + forecasting under uncertainty
        """

        if conf_int not in self.conf_int :
            raise ValueError('Confidence interval selected not in conf_int_list')

        self._plot_Y_w_uncertainty(xlim = xlim, ylim = ylim, title = title, conf_int = conf_int, save = save)

        # Plot components of x_DF
        for ts_component in list(self.x_names):
            self._plot_component_w_uncertainty(xlim, ylim, title, ts_component, conf_int, save)

        return