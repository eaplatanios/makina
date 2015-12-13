package org.platanios.learn.classification.active;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.statistics.StatisticsUtilities;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum ActiveLearningMethod {
    RANDOM {
        @Override
        public <V extends Vector> Learning.InstanceToLabel<V> pickInstanceToLabel(
                Learning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets
        ) {
            return pickInstancesToLabel(learning, dataSets, 1).get(0);
        }

        @Override
        public <V extends Vector> List<Learning.InstanceToLabel<V>> pickInstancesToLabel(
                Learning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets,
                int numberOfInstancesToPick
        ) {
            List<Learning.InstanceToLabel<V>> instances = collectInstances(dataSets);
            return StatisticsUtilities.sampleWithoutReplacement(instances,
                                                                Math.min(numberOfInstancesToPick, instances.size()));
        }

        private <V extends Vector> List<Learning.InstanceToLabel<V>> collectInstances(
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets
        ) {
            List<Learning.InstanceToLabel<V>> instances = new ArrayList<>();
            for (Map.Entry<Label, DataSet<PredictedDataInstance<V, Double>>> instanceEntry : dataSets.entrySet()) {
                DataSet<PredictedDataInstance<V, Double>> currentDataSet = instanceEntry.getValue();
                for (PredictedDataInstance<V, Double> instance : currentDataSet)
                    instances.add(new Learning.InstanceToLabel<>(instanceEntry.getKey(), instance));
            }
            return instances;
        }
    },
    UNCERTAINTY_HEURISTIC {
        @Override
        public <V extends Vector> Learning.InstanceToLabel<V> pickInstanceToLabel(
                Learning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets
        ) {
            Set<Map.Entry<Learning.InstanceToLabel<V>, Double>> instanceEntropies =
                    computeInstanceEntropies(dataSets).entrySet();
            if (instanceEntropies.size() > 0)
                return instanceEntropies
                        .stream()
                        .max(Map.Entry.comparingByValue(Double::compareTo))
                        .get()
                        .getKey();
            else
                return null;
        }

        @Override
        public <V extends Vector> List<Learning.InstanceToLabel<V>> pickInstancesToLabel(
                Learning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets,
                int numberOfInstancesToPick
        ) {
            Set<Map.Entry<Learning.InstanceToLabel<V>, Double>> instanceEntropies =
                    computeInstanceEntropies(dataSets).entrySet();
            if (instanceEntropies.size() > 0)
                return instanceEntropies
                        .stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue(Double::compareTo)))
                        .collect(Collectors.toList())
                        .subList(0, Math.min(numberOfInstancesToPick, instanceEntropies.size()))
                        .stream()
                        .map(Map.Entry::getKey)
                        .collect(Collectors.toList());
            else
                return new ArrayList<>();
        }

        private <V extends Vector> Map<Learning.InstanceToLabel<V>, Double> computeInstanceEntropies(
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets
        ) {
            Map<Learning.InstanceToLabel<V>, Double> instanceEntropies = new HashMap<>();
            for (Map.Entry<Label, DataSet<PredictedDataInstance<V, Double>>> instanceEntry : dataSets.entrySet()) {
                DataSet<PredictedDataInstance<V, Double>> currentDataSet = instanceEntry.getValue();
                for (PredictedDataInstance<V, Double> instance : currentDataSet)
                    if (instance.label() > 0.5)
                        instanceEntropies.put(new Learning.InstanceToLabel<>(instanceEntry.getKey(), instance),
                                              entropy(instance.probability()));
                    else
                        instanceEntropies.put(new Learning.InstanceToLabel<>(instanceEntry.getKey(), instance),
                                              entropy(1 - instance.probability()));
            }
            return instanceEntropies;
        }
    },
    CONSTRAINT_PROPAGATION_HEURISTIC {
        @Override
        public <V extends Vector> Learning.InstanceToLabel<V> pickInstanceToLabel(
                Learning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets
        ) {
            if (!(learning instanceof ConstrainedLearning))
                throw new IllegalArgumentException("This active learning method can only " +
                                                           "be used with the constrained learner.");



            return null;
        }

        @Override
        public <V extends Vector> List<Learning.InstanceToLabel<V>> pickInstancesToLabel(
                Learning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets,
                int numberOfInstancesToPick
        ) {
            return null;
        }
    };

    public abstract <V extends Vector> Learning.InstanceToLabel<V> pickInstanceToLabel(
            Learning<V> learning,
            Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets
    );

    public abstract <V extends Vector> List<Learning.InstanceToLabel<V>> pickInstancesToLabel(
            Learning<V> learning,
            Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets,
            int numberOfInstancesToPick
    );

    protected double entropy(double probability) {
        if (probability > 0)
            return -probability * Math.log(probability) - (1 - probability) * Math.log(1 - probability);
        else
            return 0;
    }
}
