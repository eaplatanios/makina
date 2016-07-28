import integrator_pb2
import os


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
        load_error_rates_from_protobin(error_rates, filename)
    elif file_extension == 'csv':
        load_error_rates_from_csv(error_rates, filename)
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
        f.write(str(error_rate[0]) + ',' + error_rate[1] + ',' + str(error_rate[2]) + '\n')
    f.close()


def load_error_rates_from_csv(error_rates, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'LABEL,FUNCTION_ID,VALUE':
            continue
        error_rates.append((int(line[0]), line[1], float(line[2])))
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
        load_predicted_instances_from_protobin(predicted_instances, filename)
    elif file_extension == 'csv':
        load_predicted_instances_from_csv(predicted_instances, filename)
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
        predicted_instances.append((int(line[0]), line[1], int(line[2]), float(line[3])))
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
        load_observed_instances_from_protobin(observed_instances, filename)
    elif file_extension == 'csv':
        load_observed_instances_from_csv(observed_instances, filename)
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
        observed_instances.append((int(line[0]), line[1], line[2] == '1'))
    return observed_instances


def get_filename_extension(filename):
    return os.path.splitext(filename)[1]
