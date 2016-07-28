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


def save_predicted_data(predicted_data, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_predicted_data_to_protobin(predicted_data, filename)
    elif file_extension == 'csv':
        save_predicted_data_to_csv(predicted_data, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_predicted_data(predicted_data, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        load_predicted_data_from_protobin(predicted_data, filename)
    elif file_extension == 'csv':
        load_predicted_data_from_csv(predicted_data, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_predicted_data_to_protobin(predicted_data, filename):
    predicted_instances = integrator_pb2.PredictedInstances()
    for instance in predicted_data:
        predicted_instance = predicted_instances.predictedInstance.add()
        predicted_instance.id = instance[0]
        predicted_instance.label = instance[1]
        predicted_instance.functionId = instance[2]
        predicted_instance.value = instance[3]
    f = open(filename, 'wb')
    f.write(predicted_instances.SerializeToString())
    f.close()


def load_predicted_data_from_protobin(predicted_data, filename):
    predicted_instances = integrator_pb2.PredictedInstances()
    f = open(filename, 'rb')
    predicted_instances.ParseFromString(f.read())
    f.close()
    for predicted_instance in predicted_instances.predictedInstance:
        predicted_data.append((predicted_instance.id,
                               predicted_instance.label,
                               predicted_instance.functionId,
                               predicted_instance.value))
    return predicted_data


def save_predicted_data_to_csv(predicted_data, filename):
    f = open(filename, 'w')
    f.write('ID,LABEL,FUNCTION_ID,VALUE\n')
    for instance in predicted_data:
        f.write(str(instance[0]) + ',' + instance[1] + ',' + str(instance[2]) + ',' + str(instance[3]) + '\n')
    f.close()


def load_predicted_data_from_csv(predicted_data, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'ID,LABEL,FUNCTION_ID,VALUE':
            continue
        predicted_data.append((int(line[0]), line[1], int(line[2]), float(line[3])))
    return predicted_data


def save_observed_data(observed_data, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        save_observed_data_to_protobin(observed_data, filename)
    elif file_extension == 'csv':
        save_observed_data_to_csv(observed_data, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def load_observed_data(observed_data, filename):
    file_extension = get_filename_extension(filename)
    if file_extension == 'protobin':
        load_observed_data_from_protobin(observed_data, filename)
    elif file_extension == 'csv':
        load_observed_data_from_csv(observed_data, filename)
    else:
        raise ValueError('Unsupported file extension: ' + file_extension + '.')


def save_observed_data_to_protobin(observed_data, filename):
    observed_instances = integrator_pb2.ObservedInstances()
    for instance in observed_data:
        observed_instance = observed_instances.observedInstance.add()
        observed_instance.id = instance[0]
        observed_instance.label = instance[1]
        observed_instance.value = instance[2]
    f = open(filename, 'wb')
    f.write(observed_instances.SerializeToString())
    f.close()


def load_observed_data_from_protobin(observed_data, filename):
    observed_instances = integrator_pb2.ObservedInstances()
    f = open(filename, 'rb')
    observed_instances.ParseFromString(f.read())
    f.close()
    for observed_instance in observed_instances.observedInstance:
        observed_data.append((observed_instance.id,
                              observed_instance.label,
                              observed_instance.functionId,
                              observed_instance.value))
    return observed_data


def save_observed_data_to_csv(observed_data, filename):
    f = open(filename, 'w')
    f.write('ID,LABEL,VALUE\n')
    for instance in observed_data:
        f.write(str(instance[0]) + ',' + instance[1] + ',' + ('1' if instance[2] else '0') + '\n')
    f.close()


def load_observed_data_from_csv(observed_data, filename):
    f = open(filename, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    for line in lines:
        if line == 'ID,LABEL,VALUE':
            continue
        observed_data.append((int(line[0]), line[1], line[2] == '1'))
    return observed_data


def get_filename_extension(filename):
    return os.path.splitext(filename)[1]
