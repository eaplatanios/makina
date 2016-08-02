import abc

__author__ = 'Emmanouil Antonios Platanios'


class Constraint(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def from_string(string):
        if string[0] != '!':
            return MutualExclusionConstraint.from_string(string)
        else:
            return SubsumptionConstraint.from_string(string)

    @abc.abstractmethod
    def __str__(self):
        pass


class MutualExclusionConstraint(Constraint):
    def __init__(self, labels):
        self.labels = labels

    @staticmethod
    def from_string(string):
        return MutualExclusionConstraint(string[1:].split(','))

    def __str__(self):
        return '!' + ','.join(self.labels)


class SubsumptionConstraint(Constraint):
    def __init__(self, parent_label, children_labels):
        self.parent_label = parent_label
        self.children_labels = children_labels

    @staticmethod
    def from_string(string):
        string_parts = string.split(' -> ')
        return SubsumptionConstraint(string_parts[0], string_parts[1].split(','))

    def __str__(self):
        return self.parent_label + ' -> ' + ','.join(self.children_labels)
