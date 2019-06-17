from meta_policy_search.baselines.base import Baseline
from meta_policy_search.utils.serializable import Serializable
import numpy as np


class LinearBaseline(Baseline):
    """
    Abstract class providing the functionality for fitting a linear baseline
    Don't instantiate this class. Instead use LinearFeatureBaseline or LinearTimeBaseline
    """

    def __init__(self, reg_coeff=1e-5):
        super(LinearBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def predict(self, path):
        """
        Abstract Class for the LinearFeatureBaseline and the LinearTimeBaseline
        Predicts the linear reward baselines estimates for a provided trajectory / path.
        If the baseline is not fitted - returns zero baseline

        Args:
           path (dict): dict of lists/numpy array containing trajectory / path information
                 such as "observations", "rewards", ...

        Returns:
             (np.ndarray): numpy array of the same length as paths["observations"] specifying the reward baseline

        """
        if self._coeffs is None:
            return np.zeros(len(path["observations"]))
        return self._features(path).dot(self._coeffs)

    def get_param_values(self, **tags):
        """
        Returns the parameter values of the baseline object

        Returns:
            numpy array of linear_regression coefficients

        """
        return self._coeffs

    def set_params(self, value, **tags):
        """
        Sets the parameter values of the baseline object

        Args:
            value: numpy array of linear_regression coefficients

        """
        self._coeffs = value

    def fit(self, paths, target_key='returns'):
        """
        Fits the linear baseline model with the provided paths via damped least squares

        Args:
            paths (list): list of paths
            target_key (str): path dictionary key of the target that shall be fitted (e.g. "returns")

        """
        assert all([target_key in path.keys() for path in paths])

        featmat = np.concatenate([self._features(path) for path in paths], axis=0)
        target = np.concatenate([path[target_key] for path in paths], axis=0)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(target),
                rcond=-1
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def _features(self, path):
        raise NotImplementedError("this is an abstract class, use either LinearFeatureBaseline or LinearTimeBaseline")


class LinearFeatureBaseline(LinearBaseline):
    """
    Linear (polynomial) time-state dependent return baseline model
    (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)

    Fits the following linear model

    reward = b0 + b1*obs + b2*obs^2 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """
    def __init__(self, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def _features(self, path):
        obs = np.clip(path["observations"], -10, 10)
        path_length = len(path["observations"])
        time_step = np.arange(path_length).reshape(-1, 1) / 100.0
        return np.concatenate([obs, obs ** 2, time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)


class LinearTimeBaseline(LinearBaseline):
    """
    Linear (polynomial) time-dependent reward baseline model

    Fits the following linear model

    reward = b0 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """

    def _features(self, path):
        path_length = len(path["observations"])
        time_step = np.arange(path_length).reshape(-1, 1) / 100.0
        return np.concatenate([time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)





# from meta_policy_search.baselines.base import Baseline
# from meta_policy_search.utils.serializable import Serializable
# import numpy as np
import tensorflow as tf


# class LinearBaseline(Baseline):
#     """
#     Abstract class providing the functionality for fitting a linear baseline
#     Don't instantiate this class. Instead use LinearFeatureBaseline or LinearTimeBaseline
#     """

#     def __init__(self, reg_coeff=1e-5):
#         super(LinearBaseline, self).__init__()
#         self._coeffs = None
#         self._reg_coeff = reg_coeff

#     def predict(self, path):
#         """
#         Abstract Class for the LinearFeatureBaseline and the LinearTimeBaseline
#         Predicts the linear reward baselines estimates for a provided trajectory / path.
#         If the baseline is not fitted - returns zero baseline

#         Args:
#            path (dict): dict of lists/numpy array containing trajectory / path information
#                  such as "observations", "rewards", ...

#         Returns:
#              (np.ndarray): numpy array of the same length as paths["observations"] specifying the reward baseline

#         """
#         if self._coeffs is None:
#             return np.zeros(len(path["observations"]))
#         return self._features(path).dot(self._coeffs)

#     def get_param_values(self, **tags):
#         """
#         Returns the parameter values of the baseline object

#         Returns:
#             numpy array of linear_regression coefficients

#         """
#         return self._coeffs

#     def set_params(self, value, **tags):
#         """
#         Sets the parameter values of the baseline object

#         Args:
#             value: numpy array of linear_regression coefficients

#         """
#         self._coeffs = value

#     def fit(self, paths, target_key='returns'):
#         """
#         Fits the linear baseline model with the provided paths via damped least squares

#         Args:
#             paths (list): list of paths
#             target_key (str): path dictionary key of the target that shall be fitted (e.g. "returns")

#         """
#         assert all([target_key in path.keys() for path in paths])

#         featmat = np.concatenate([self._features(path)
#                                   for path in paths], axis=0)
#         target = np.concatenate([path[target_key] for path in paths], axis=0)
#         reg_coeff = self._reg_coeff
#         for _ in range(5):
#             self._coeffs = np.linalg.lstsq(
#                 featmat.T.dot(featmat) + reg_coeff *
#                 np.identity(featmat.shape[1]),
#                 featmat.T.dot(target),
#                 rcond=-1
#             )[0]
#             if not np.any(np.isnan(self._coeffs)):
#                 break
#             reg_coeff *= 10

#     def _features(self, path):
#         raise NotImplementedError(
#             "this is an abstract class, use either LinearFeatureBaseline or LinearTimeBaseline")


# class LinearFeatureBaseline(LinearBaseline):
#     """
#     Linear (polynomial) time-state dependent return baseline model
#     (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)

#     Fits the following linear model

#     reward = b0 + b1*obs + b2*obs^2 + b3*t + b4*t^2+  b5*t^3

#     Args:
#         reg_coeff: list of paths

#     """

#     def __init__(self, reg_coeff=1e-5):
#         super(LinearFeatureBaseline, self).__init__()
#         self._coeffs = None
#         self._reg_coeff = reg_coeff

#     def _features(self, path):
#         obs = np.clip(path["observations"], -10, 10)
#         path_length = len(path["observations"])
#         time_step = np.arange(path_length).reshape(-1, 1) / 100.0
#         return np.concatenate([obs, obs ** 2, time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
#                               axis=1)


# class LinearTimeBaseline(LinearBaseline):
#     """
#     Linear (polynomial) time-dependent reward baseline model

#     Fits the following linear model

#     reward = b0 + b3*t + b4*t^2+  b5*t^3

#     Args:
#         reg_coeff: list of paths

#     """

#     def _features(self, path):
#         path_length = len(path["observations"])
#         time_step = np.arange(path_length).reshape(-1, 1) / 100.0
#         return np.concatenate([time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
#                               axis=1)


class Network:
    def __init__(self, innerstepsize, input_size, output_size, mode):
        self.innerstepsize = innerstepsize
        self.mode = mode
        self.input_size = input_size
        self.output_size = output_size
        self.build()

    def get_param_values(self, **tags):
        raise NotImplementedError()

    def set_params(self, value, **tags):
        raise NotImplementedError()

    def forward(self, x, scope):
        with tf.variable_scope(scope):
            output = tf.layers.dense(x, 64)
            output = tf.nn.tanh(output)
            output = tf.layers.dense(output, 64)
            output = tf.nn.tanh(output)
            y = tf.layers.dense(output, self.output_size, name='y')
        return y

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_size], name='x')
        self.y = self.forward(self.x, self.mode)

        _ = self.forward(self.x, 'backup')

        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.mode)
        backup_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'backup')

        self.label = tf.placeholder(
            tf.float32, [None, self.output_size], name='label')
        self.loss = tf.reduce_mean(tf.square(self.y - self.label))
        self.gradients = tf.gradients(self.loss, variables)

        inner_optimize_expr = []
        for var, grad in zip(variables, self.gradients):
            inner_optimize_expr.append(
                var.assign(var - self.innerstepsize * grad))
        self.inner_optimize = tf.group(*inner_optimize_expr)

        backup_expr = []
        for var, backup_var in zip(variables, backup_variables):
            backup_expr.append(backup_var.assign(var))
        self.backup_ops = tf.group(*backup_expr)

        restore_expr = []
        for var, backup_var in zip(variables, backup_variables):
            restore_expr.append(var.assign(backup_var))
        self.restore_ops = tf.group(*restore_expr)

        self.outerstepsize = tf.placeholder(
            tf.float32, [], name='outerstepsize')

        outer_optimize_expr = []
        if self.mode == 'reptile':
            for var, backup_var in zip(variables, backup_variables):
                outer_optimize_expr.append(
                    var.assign(backup_var + self.outerstepsize * (var - backup_var)))
        elif self.mode == 'maml':
            for var, backup_var, grad in zip(variables, backup_variables, self.gradients):
                outer_optimize_expr.append(
                    var.assign(backup_var - self.outerstepsize * grad))
        self.outer_optimize = tf.group(*outer_optimize_expr)

    def get_session(self):
        return tf.get_default_session()

    def predict(self, x):
        sess = self.get_session()
        with sess.as_default():
            return sess.run(self.y, feed_dict={self.x: x})

    def inner_update(self, x, label):
        sess = self.get_session()
        with sess.as_default():
            feed_dict = {
                self.x: x,
                self.label: label
            }
            loss, _ = sess.run(
                [self.loss, self.inner_optimize], feed_dict=feed_dict)
        return loss

    def outer_update(self, x, label, outerstepsize):
        sess = self.get_session()
        with sess.as_default():
            feed_dict = {
                self.x: x,
                self.label: label,
                self.outerstepsize: outerstepsize
            }
            sess.run([self.gradients, self.outer_optimize],
                     feed_dict=feed_dict)

    def backup(self):
        sess = self.get_session()
        with sess.as_default():
            sess.run(self.backup_ops)

    def restore(self):
        sess = self.get_session()
        with sess.as_default():
            sess.run(self.restore_ops)


class MetaNNBaseline(Baseline):
    def __init__(self, input_size, output_size=1, mode='maml', innerstepsize=2e-2, outerstepsize=1e-3,
                 innerepochs=1, niterations=5, step=100):
        super(MetaNNBaseline, self).__init__()
        self.mode = mode
        self.outerstepsize = outerstepsize
        self.niterations = niterations
        self.innerepochs = innerepochs
        self.model = Network(innerstepsize=innerstepsize,
                             input_size=input_size, output_size=output_size, mode=mode)
        self.step = step

    def train_on_batch(self, path, returns):
        self.model.inner_update(path, returns)

    def predict(self, path):
        return np.squeeze(self.model.predict(path["observations"]), axis=-1)

    def fit(self, paths, target_key='returns'):
        assert all([target_key in path.keys() for path in paths])
        x_all = np.concatenate(
            [np.clip(path["observations"], -10, 10) for path in paths], axis=0)
        y_all = np.concatenate([path[target_key] for path in paths], axis=0)
        if y_all.ndim < 2:
            y_all = np.expand_dims(y_all, axis=1)
        for _ in range(self.niterations):
            self.model.backup()
            inds = np.random.permutation(len(x_all))
            for _ in range(self.innerepochs):
                for start in range(0, x_all.shape[0], self.step):
                    mbinds = inds[start:start+self.step]
                    self.train_on_batch(x_all[mbinds, :], y_all[mbinds])
            self.model.outer_update(x_all, y_all, self.outerstepsize)
            self.model.backup()