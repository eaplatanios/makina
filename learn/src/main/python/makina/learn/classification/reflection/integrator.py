import abc
import errno
import integrator_pb2
import os
import subprocess

from makina.learn.classification.constraint import Constraint
from makina.utilities import logger

__author__ = 'Emmanouil Antonios Platanios'


def silently_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # This error code corresponds to the file not having been found and is ignored here
            raise                    # If the actual error was different we re-raise the exception


class Integrator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def options(self):
        pass

    @abc.abstractmethod
    def build_java_object(self, predicted_data, observed_data=None, constraints=None, seed=None):
        pass

    def run(self, predicted_data, observed_data=None, constraints=None, integrate_data=False, seed=None,
            use_command_line_interface=False, working_directory='.', makina_jar='./nig/evaluation/makina.jar',
            use_csv=False, clean_up=True, jvm_options=None):
        if not use_command_line_interface:
            return self._run_jnius(predicted_data, observed_data, constraints, integrate_data, seed, makina_jar,
                                   jvm_options)
        else:
            return self._run_command_line(predicted_data, observed_data, constraints, integrate_data, seed,
                                          working_directory, makina_jar, use_csv, clean_up, jvm_options)

    def _run_jnius(self, predicted_data, observed_data, constraints, integrate_data, seed, makina_jar, jvm_options):
        if jvm_options is None:
            jvm_options = ['-Xmx4G']
        import jnius_config
        try:
            jnius_config.add_options(*jvm_options)
            jnius_config.set_classpath(makina_jar)
        except ValueError:
            pass
        if observed_data is None:
            integrator = self.build_java_object(_convert_predicted_instances_to_java_integrator_data(predicted_data),
                                                constraints=constraints, seed=seed)
        else:
            integrator = self.build_java_object(_convert_predicted_instances_to_java_integrator_data(predicted_data),
                                                _convert_observed_instances_to_java_integrator_data(observed_data),
                                                constraints=constraints, seed=seed)
        error_rates = _convert_java_error_rates_to_error_rates(integrator.errorRates())
        if not integrate_data:
            return error_rates
        else:
            integrated_data = _convert_java_predicted_instances_to_predicted_instances(integrator.integratedData())
            return error_rates, integrated_data

    def _run_command_line(self, predicted_data, observed_data, constraints, integrate_data, seed, working_directory,
                          makina_jar, use_csv, clean_up, jvm_options):
        file_extension = '.protobin'
        if use_csv:
            file_extension = '.csv'
        predicted_data_file = os.path.join(working_directory, 'predicted_instances' + file_extension)
        observed_data_file = os.path.join(working_directory, 'observed_instances' + file_extension)
        constraints_file = os.path.join(working_directory, 'constraints.txt')
        error_rates_file = os.path.join(working_directory, 'error_rates' + file_extension)
        integrated_data_file = os.path.join(working_directory, 'integrated_data' + file_extension)
        save_predicted_instances(predicted_data, predicted_data_file)
        if observed_data is not None:
            save_observed_instances(observed_data, observed_data_file)
        if jvm_options is None:
            jvm_options = ['-Xmx4G']
        command_line_options = ['java', '-cp', makina_jar]
        command_line_options.extend(jvm_options)
        command_line_options.extend(['makina.learn.classification.reflection.Integrator', '-d', predicted_data_file,
                                     '-e', error_rates_file, '-m', self.name()])
        if self.options():
            command_line_options.extend(['-o', ':'.join(self.options())])
        if constraints is not None:
            save_constraints(constraints, constraints_file)
            command_line_options.extend(['-c', constraints_file])
        if integrate_data:
            command_line_options.extend(['-i', integrated_data_file])
        if seed is not None:
            command_line_options.extend(['-s', seed])
        return_code = subprocess.Popen(command_line_options).wait()
        if return_code != 0:
            logger.error('An error occurred while running the command "' + ' '.join(command_line_options) + '".')
            return None
        error_rates = load_error_rates([], error_rates_file)
        integrated_data = load_predicted_instances([], integrated_data_file) if integrate_data else None
        if clean_up:
            silently_remove(predicted_data_file)
            silently_remove(observed_data_file)
            silently_remove(constraints_file)
            silently_remove(error_rates_file)
            silently_remove(integrated_data_file)
        if integrate_data:
            return error_rates, integrated_data
        else:
            return error_rates


class MajorityVoteIntegrator(Integrator):
    def name(self):
        return 'MVI'

    def options(self):
        return []

    def build_java_object(self, predicted_data, observed_data=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass('makina.learn.classification.reflection.MajorityVoteIntegrator$Builder')
        return builder(predicted_data).build()


class AgreementIntegrator(Integrator):
    def __init__(self, highest_order=-1, only_even_cardinality_subsets_agreements=True):
        self.highest_order = highest_order
        self.only_even_cardinality_subsets_agreements = only_even_cardinality_subsets_agreements

    def name(self):
        return 'AI'

    def options(self):
        return [str(self.highest_order), '1' if self.only_even_cardinality_subsets_agreements else '0']

    def build_java_object(self, predicted_data, observed_data=None, constraints=None, seed=None):
        pass


class BayesianIntegrator(Integrator):
    def __init__(self, number_of_burn_in_samples=4000, number_of_thinning_samples=10, number_of_samples=200,
                 labels_prior_alpha=1.0, labels_prior_beta=1.0, error_rates_prior_alpha=1.0,
                 error_rates_prior_beta=2.0):
        self.number_of_burn_in_samples = number_of_burn_in_samples
        self.number_of_thinning_samples = number_of_thinning_samples
        self.number_of_samples = number_of_samples
        self.labels_prior_alpha = labels_prior_alpha
        self.labels_prior_beta = labels_prior_beta
        self.error_rates_prior_alpha = error_rates_prior_alpha
        self.error_rates_prior_beta = error_rates_prior_beta

    def name(self):
        return 'BI'

    def options(self):
        return [str(self.number_of_burn_in_samples), str(self.number_of_thinning_samples), str(self.number_of_samples),
                str(self.labels_prior_alpha), str(self.labels_prior_beta), str(self.error_rates_prior_alpha),
                str(self.error_rates_prior_beta)]

    def build_java_object(self, predicted_data, observed_data=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass('makina.learn.classification.reflection.BayesianIntegrator$Builder')
        return builder(predicted_data) \
            .numberOfBurnInSamples(self.number_of_burn_in_samples) \
            .numberOfThinningSamples(self.number_of_thinning_samples) \
            .numberOfSamples(self.number_of_samples) \
            .labelsPriorAlpha(self.labels_prior_alpha) \
            .labelsPriorBeta(self.labels_prior_beta) \
            .errorRatesPriorAlpha(self.error_rates_prior_alpha) \
            .errorRatesPriorBeta(self.error_rates_prior_beta) \
            .seed(seed) \
            .build()


class CoupledBayesianIntegrator(Integrator):
    def __init__(self, number_of_burn_in_samples=4000, number_of_thinning_samples=10, number_of_samples=200, alpha=1.0,
                 labels_prior_alpha=1.0, labels_prior_beta=1.0, error_rates_prior_alpha=1.0,
                 error_rates_prior_beta=2.0):
        self.number_of_burn_in_samples = number_of_burn_in_samples
        self.number_of_thinning_samples = number_of_thinning_samples
        self.number_of_samples = number_of_samples
        self.alpha = alpha
        self.labels_prior_alpha = labels_prior_alpha
        self.labels_prior_beta = labels_prior_beta
        self.error_rates_prior_alpha = error_rates_prior_alpha
        self.error_rates_prior_beta = error_rates_prior_beta

    def name(self):
        return 'CBI'

    def options(self):
        return [str(self.number_of_burn_in_samples), str(self.number_of_thinning_samples), str(self.number_of_samples),
                str(self.alpha), str(self.labels_prior_alpha), str(self.labels_prior_beta),
                str(self.error_rates_prior_alpha), str(self.error_rates_prior_beta)]

    def build_java_object(self, predicted_data, observed_data=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass('makina.learn.classification.reflection.CoupledBayesianIntegrator$Builder')
        return builder(predicted_data) \
            .numberOfBurnInSamples(self.number_of_burn_in_samples) \
            .numberOfThinningSamples(self.number_of_thinning_samples) \
            .numberOfSamples(self.number_of_samples) \
            .alpha(self.alpha) \
            .labelsPriorAlpha(self.labels_prior_alpha) \
            .labelsPriorBeta(self.labels_prior_beta) \
            .errorRatesPriorAlpha(self.error_rates_prior_alpha) \
            .errorRatesPriorBeta(self.error_rates_prior_beta) \
            .seed(seed) \
            .build()


class HierarchicalCoupledBayesianIntegrator(Integrator):
    def __init__(self, number_of_burn_in_samples=4000, number_of_thinning_samples=10, number_of_samples=200, alpha=1.0,
                 gamma=1.0, labels_prior_alpha=1.0, labels_prior_beta=1.0, error_rates_prior_alpha=1.0,
                 error_rates_prior_beta=2.0):
        self.number_of_burn_in_samples = number_of_burn_in_samples
        self.number_of_thinning_samples = number_of_thinning_samples
        self.number_of_samples = number_of_samples
        self.alpha = alpha
        self.gamma = gamma
        self.labels_prior_alpha = labels_prior_alpha
        self.labels_prior_beta = labels_prior_beta
        self.error_rates_prior_alpha = error_rates_prior_alpha
        self.error_rates_prior_beta = error_rates_prior_beta

    def name(self):
        return 'HCBI'

    def options(self):
        return [str(self.number_of_burn_in_samples), str(self.number_of_thinning_samples), str(self.number_of_samples),
                str(self.alpha), str(self.gamma), str(self.labels_prior_alpha), str(self.labels_prior_beta),
                str(self.error_rates_prior_alpha), str(self.error_rates_prior_beta)]

    def build_java_object(self, predicted_data, observed_data=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass('makina.learn.classification.reflection.HierarchicalCoupledBayesianIntegrator$Builder')
        return builder(predicted_data) \
            .numberOfBurnInSamples(self.number_of_burn_in_samples) \
            .numberOfThinningSamples(self.number_of_thinning_samples) \
            .numberOfSamples(self.number_of_samples) \
            .alpha(self.alpha) \
            .gamma(self.gamma) \
            .labelsPriorAlpha(self.labels_prior_alpha) \
            .labelsPriorBeta(self.labels_prior_beta) \
            .errorRatesPriorAlpha(self.error_rates_prior_alpha) \
            .errorRatesPriorBeta(self.error_rates_prior_beta) \
            .seed(seed) \
            .build()


class LogicIntegrator(Integrator):
    def name(self):
        return 'LI'

    def options(self):
        return []

    def build_java_object(self, predicted_data, observed_data=None, constraints=None, seed=None):
        from jnius import autoclass
        builder = autoclass('makina.learn.classification.reflection.LogicIntegrator$Builder')
        hash_set_class = autoclass('java.util.HashSet')
        java_constraints = hash_set_class()
        if constraints is not None:
            for constraint in constraints:
                java_constraints.add(constraint.build_java_object())
        return builder(predicted_data, observed_data).addConstraints(java_constraints).build()


def _convert_predicted_instances_to_java_integrator_data(predicted_instances):
    from jnius import autoclass
    array_list_class = autoclass('java.util.ArrayList')
    instances = array_list_class()
    label_class = autoclass('makina.learn.classification.Label')
    predicted_instance_class = autoclass('makina.learn.classification.reflection.Integrator$Data$PredictedInstance')
    for instance in predicted_instances:
        instances.add(predicted_instance_class(instance[0], label_class(instance[1]), instance[2], instance[3]))
    return autoclass('makina.learn.classification.reflection.Integrator$Data')(instances)


def _convert_observed_instances_to_java_integrator_data(predicted_instances):
    from jnius import autoclass
    array_list_class = autoclass('java.util.ArrayList')
    instances = array_list_class()
    label_class = autoclass('makina.learn.classification.Label')
    observed_instance_class = autoclass('makina.learn.classification.reflection.Integrator$Data$ObservedInstance')
    for instance in predicted_instances:
        instances.add(observed_instance_class(instance[0], label_class(instance[1]), instance[2]))
    return autoclass('makina.learn.classification.reflection.Integrator$Data')(instances)


def _iterate_java_iterator(java_iterator):
    while java_iterator.hasNext():
        yield java_iterator.next()


def _convert_java_error_rates_to_error_rates(java_error_rates):
    error_rates = []
    for java_error_rate in _iterate_java_iterator(java_error_rates.iterator()):
        error_rates.append((java_error_rate.label().name(), java_error_rate.functionId(), java_error_rate.value()))
    return error_rates


def _convert_java_predicted_instances_to_predicted_instances(java_predicted_instances):
    predicted_instances = []
    for java_predicted_instance in _iterate_java_iterator(java_predicted_instances.iterator()):
        predicted_instances.append((java_predicted_instance.id(),
                                    java_predicted_instance.label().name(),
                                    java_predicted_instance.functionId(),
                                    java_predicted_instance.value()))
    return predicted_instances


def _convert_java_observed_instances_to_observed_instances(java_observed_instances):
    observed_instances = []
    for java_observed_instance in _iterate_java_iterator(java_observed_instances.iterator()):
        observed_instances.append((java_observed_instance.id(), java_observed_instance.label().name(),
                                   java_observed_instance.value()))
    return observed_instances


def save_error_rates(error_rates, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_error_rates_to_protobin(error_rates, filename)
    elif file_extension == 'csv':
        save_error_rates_to_csv(error_rates, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_error_rates(error_rates, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        return load_error_rates_from_protobin(error_rates, filename)
    elif file_extension == 'csv':
        return load_error_rates_from_csv(error_rates, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_error_rates_to_protobin(error_rates, filename):
    proto_error_rates = integrator_pb2.ErrorRates()
    for error_rate in error_rates:
        proto_error_rate = proto_error_rates.errorRate.add()
        proto_error_rate.label = error_rate[0]
        proto_error_rate.functionId = error_rate[1]
        proto_error_rate.value = error_rate[2]
    f = open(filename, 'wb')
    f.write(proto_error_rates.SerializeToString())
    f.close()


def load_error_rates_from_protobin(error_rates, filename):
    proto_error_rates = integrator_pb2.ErrorRates()
    f = open(filename, 'rb')
    proto_error_rates.ParseFromString(f.read())
    f.close()
    for proto_error_rate in proto_error_rates.errorRate:
        error_rates.append((proto_error_rate.label, proto_error_rate.functionId, proto_error_rate.value))
    return error_rates


def save_error_rates_to_csv(error_rates, filename):
    f = open(filename, 'w')
    f.write('LABEL,FUNCTION_ID,VALUE\n')
    for error_rate in error_rates:
        f.write(error_rate[0] + ',' + str(error_rate[1]) + ',' + str(error_rate[2]) + '\n')
    f.close()


def load_error_rates_from_csv(error_rates, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'LABEL,FUNCTION_ID,VALUE':
            continue
        line_parts = line.split(',')
        error_rates.append((line_parts[0], int(line_parts[1]), float(line_parts[2])))
    return error_rates


def save_predicted_instances(predicted_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_predicted_instances_to_protobin(predicted_instances, filename)
    elif file_extension == 'csv':
        save_predicted_instances_to_csv(predicted_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_predicted_instances(predicted_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        return load_predicted_instances_from_protobin(predicted_instances, filename)
    elif file_extension == 'csv':
        return load_predicted_instances_from_csv(predicted_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_predicted_instances_to_protobin(predicted_instances, filename):
    proto_predicted_instances = integrator_pb2.PredictedInstances()
    for predicted_instance in predicted_instances:
        proto_predicted_instance = proto_predicted_instances.predictedInstance.add()
        proto_predicted_instance.id = predicted_instance[0]
        proto_predicted_instance.label = predicted_instance[1]
        proto_predicted_instance.functionId = predicted_instance[2]
        proto_predicted_instance.value = predicted_instance[3]
    f = open(filename, 'wb')
    f.write(proto_predicted_instances.SerializeToString())
    f.close()


def load_predicted_instances_from_protobin(predicted_instances, filename):
    proto_predicted_instances = integrator_pb2.PredictedInstances()
    f = open(filename, 'rb')
    proto_predicted_instances.ParseFromString(f.read())
    f.close()
    for proto_predicted_instance in proto_predicted_instances.predictedInstance:
        predicted_instances.append((proto_predicted_instance.id,
                                    proto_predicted_instance.label,
                                    proto_predicted_instance.functionId,
                                    proto_predicted_instance.value))
    return predicted_instances


def save_predicted_instances_to_csv(predicted_instances, filename):
    f = open(filename, 'w')
    f.write('ID,LABEL,FUNCTION_ID,VALUE\n')
    for instance in predicted_instances:
        f.write(str(instance[0]) + ',' + instance[1] + ',' + str(instance[2]) + ',' + str(instance[3]) + '\n')
    f.close()


def load_predicted_instances_from_csv(predicted_instances, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'ID,LABEL,FUNCTION_ID,VALUE':
            continue
        line_parts = line.split(',')
        predicted_instances.append((int(line_parts[0]), line_parts[1], int(line_parts[2]), float(line_parts[3])))
    return predicted_instances


def save_observed_instances(observed_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_observed_instances_to_protobin(observed_instances, filename)
    elif file_extension == 'csv':
        save_observed_instances_to_csv(observed_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_observed_instances(observed_instances, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        return load_observed_instances_from_protobin(observed_instances, filename)
    elif file_extension == 'csv':
        return load_observed_instances_from_csv(observed_instances, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_observed_instances_to_protobin(observed_instances, filename):
    proto_observed_instances = integrator_pb2.ObservedInstances()
    for observed_instance in observed_instances:
        proto_observed_instance = proto_observed_instances.observedInstance.add()
        proto_observed_instance.id = observed_instance[0]
        proto_observed_instance.label = observed_instance[1]
        proto_observed_instance.value = observed_instance[2]
    f = open(filename, 'wb')
    f.write(proto_observed_instances.SerializeToString())
    f.close()


def load_observed_instances_from_protobin(observed_instances, filename):
    proto_observed_instances = integrator_pb2.ObservedInstances()
    f = open(filename, 'rb')
    proto_observed_instances.ParseFromString(f.read())
    f.close()
    for proto_observed_instance in proto_observed_instances.observedInstance:
        observed_instances.append((proto_observed_instance.id,
                                   proto_observed_instance.label,
                                   proto_observed_instance.functionId,
                                   proto_observed_instance.value))
    return observed_instances


def save_observed_instances_to_csv(observed_instances, filename):
    f = open(filename, 'w')
    f.write('ID,LABEL,VALUE\n')
    for instance in observed_instances:
        f.write(str(instance[0]) + ',' + instance[1] + ',' + ('1' if instance[2] else '0') + '\n')
    f.close()


def load_observed_instances_from_csv(observed_instances, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'ID,LABEL,VALUE':
            continue
        line_parts = line.split(',')
        observed_instances.append((int(line_parts[0]), line_parts[1], line_parts[2] == '1'))
    return observed_instances


def get_filename_extension(filename):
    return os.path.splitext(filename)[1][1:]


def save_constraints(constraints, filename):
    f = open(filename, 'w')
    for constraint in constraints:
        f.write(str(constraint) + '\n')
    f.close()


def load_constraints(constraints, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        parsed_constraint = Constraint.from_string(line)
        if type(parsed_constraint) is list:
            constraints.extend(parsed_constraint)
        else:
            constraints.add(parsed_constraint)
    return constraints
