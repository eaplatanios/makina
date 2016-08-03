import itertools
import numpy as np

from makina.learn.classification.reflection.integrator import *
from makina.utilities import logger, elapsed_timer

__author__ = 'Emmanouil Antonios Platanios'


def generate_sample_data(number_of_instances, number_of_functions,
                         number_of_labels):
    true_error_rates = np.random.beta(1.0, 2.0, size=[number_of_labels,
                                                      number_of_functions])
    true_labels = np.random.randint(2, size=[number_of_instances,
                                             number_of_labels])
    uniform_samples = np.random.uniform(0.0, 1.0, size=[number_of_instances,
                                                        number_of_labels,
                                                        number_of_functions])
    observations = np.tile(true_labels[:, :, np.newaxis],
                           [1, 1, number_of_functions])
    observations[uniform_samples < np.tile(true_error_rates[np.newaxis, :, :],
                                           [number_of_instances, 1, 1])] += 1
    observations %= 2
    predicted_data = [(int(i[0]), str(i[1]), int(i[2]), float(v))
                      for i, v in np.ndenumerate(observations)]
    true_error_rates = [(str(i[0]), int(i[1]), float(v))
                        for i, v in np.ndenumerate(true_error_rates)]
    true_data = [(int(i[0]), str(i[1]), float(v))
                 for i, v in np.ndenumerate(true_labels)]
    return predicted_data, true_error_rates, true_data


def sorted_array(dictionary):
    return np.array([t[1] for t in sorted(dictionary.items())])


def evaluate_integrator(predicted_data, true_error_rates, true_data, use_cli,
                        integrator):
    with elapsed_timer() as elapsed:
        error_rates, integrated_data = integrator.run(predicted_data,
                                                      integrate_data=True,
                                                      use_cli=use_cli,
                                                      jvm_options=['-Xmx12G'])

    true_error_rates = sorted_array(dict([('l' + t[0] + 'f' + str(t[1]), t[2])
                                          for t in true_error_rates]))
    error_rates = sorted_array(dict([('l' + t[0] + 'f' + str(t[1]), t[2])
                                     for t in error_rates]))
    true_data = sorted_array(dict([('i' + str(t[0]) + 'l' + t[1], t[2])
                                   for t in true_data]))
    integrated_data = sorted_array(dict([('i' + str(t[0]) + 'l' + t[1], t[3])
                                         for t in integrated_data]))
    mad_error = np.mean(np.abs(true_error_rates - error_rates))
    mse_error = np.mean(np.square(true_error_rates - error_rates))
    # rmse_error = np.sqrt(mse_error)
    mad_soft_label = np.mean(np.abs(true_data - integrated_data))
    mse_soft_label = np.mean(np.square(true_data - integrated_data))
    # rmse_soft_label = np.sqrt(mse_soft_label)
    integrated_data[integrated_data > 0.5] = 1.0
    integrated_data[integrated_data <= 0.5] = 0.0
    mad_hard_label = np.mean(np.abs(true_data - integrated_data))
    mse_hard_label = np.mean(np.square(true_data - integrated_data))
    # rmse_hard_label = np.sqrt(mse_hard_label)
    return elapsed(), mad_error, mse_error, mad_soft_label, mse_soft_label, \
           mad_hard_label, mse_hard_label

logger_row_format = '| {:>10} | {:>10} | {:>10} | {:>7} | {:>10} | {:>20} | ' \
                    '{:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} |'
logger_row_format = logger_row_format.format
logger.info(logger_row_format('#Instances', '#Functions', '#Labels', 'Use CLI',
                              'Integrator', 'Time (s)', 'MAD Error',
                              'MSE Error', 'MAD SLabel', 'MSE SLabel',
                              'MAD HLabel', 'MSE HLabel'))
logger.info('=' * 164)

logger_row_format = '| {:>10} | {:>10} | {:>10} | {:>7} | {:>10} | ' \
                    '{:>20.14e} | {:>10.4e} | {:>10.4e} | {:>10.4e} | ' \
                    '{:>10.4e} | {:>10.4e} | {:>10.4e} |'.format

number_of_instances_to_run = [100, 1000, 10000]
number_of_functions_to_run = [10, 100]
number_of_labels_to_run = [1, 5, 10]
use_cli_to_run = [True, False]
integrators_to_run = [MajorityVoteIntegrator(), BayesianIntegrator(),
                      LogicIntegrator()]

for options in itertools.product(number_of_instances_to_run,
                                 number_of_functions_to_run,
                                 number_of_labels_to_run,
                                 use_cli_to_run,
                                 integrators_to_run):
    eval_args = (generate_sample_data(*options[0:3]) + options[3:5])
    results = evaluate_integrator(*eval_args)
    logger.info(logger_row_format(options[0], options[1], options[2],
                                  'CLI' if options[3] else 'JAVA',
                                  options[4].name(), *results))
