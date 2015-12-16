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
            List<Learning.InstanceToLabel<V>> instances = pickInstancesToLabel(learning, dataSets, 1);
            return instances.size() > 0 ? instances.get(0) : null;
        }

        @Override
        public <V extends Vector> List<Learning.InstanceToLabel<V>> pickInstancesToLabel(
                Learning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets,
                int numberOfInstancesToPick
        ) {
            List<Learning.InstanceToLabel<V>> instances = collectInstances(dataSets);
            if (instances.size() > 0)
                return StatisticsUtilities.sampleWithoutReplacement(
                        instances,
                        Math.min(numberOfInstancesToPick, instances.size())
                );
            else
                return new ArrayList<>();
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
                    instanceEntropies.put(new Learning.InstanceToLabel<>(instanceEntry.getKey(), instance),
                                          instance.label() >= 0.5 ? entropy(instance.probability()) : entropy(1 - instance.probability()));
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
            Set<Map.Entry<Learning.InstanceToLabel<V>, Double>> instanceScores =
                    computeInstanceScores((ConstrainedLearning<V>) learning, dataSets).entrySet();
            if (instanceScores.size() > 0)
                return instanceScores
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
            if (!(learning instanceof ConstrainedLearning))
                throw new IllegalArgumentException("This active learning method can only " +
                                                           "be used with the constrained learner.");
            Set<Map.Entry<Learning.InstanceToLabel<V>, Double>> instanceScores =
                    computeInstanceScores((ConstrainedLearning<V>) learning, dataSets).entrySet();
            if (instanceScores.size() > 0)
                return instanceScores
                        .stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue(Double::compareTo)))
                        .collect(Collectors.toList())
                        .subList(0, Math.min(numberOfInstancesToPick, instanceScores.size()))
                        .stream()
                        .map(Map.Entry::getKey)
                        .collect(Collectors.toList());
            else
                return new ArrayList<>();
        }

        private <V extends Vector> Map<Learning.InstanceToLabel<V>, Double> computeInstanceScores(
                ConstrainedLearning<V> learning,
                Map<Label, DataSet<PredictedDataInstance<V, Double>>> dataSets
        ) {
//            Map<PredictedDataInstance<V, Double>, Map<Label, Double>> probabilitiesMap = new HashMap<>();
//            for (Map.Entry<Label, DataSet<PredictedDataInstance<V, Double>>> instanceEntry : dataSets.entrySet()) {
//                DataSet<PredictedDataInstance<V, Double>> currentDataSet = instanceEntry.getValue();
//                for (PredictedDataInstance<V, Double> instance : currentDataSet) {
//                    if (!probabilitiesMap.containsKey(instance))
//                        probabilitiesMap.put(instance, new HashMap<>());
//                    if (!probabilitiesMap.get(instance).containsKey(instanceEntry.getKey()))
//                        probabilitiesMap.get(instance).put(instanceEntry.getKey(), instance.label() >= 0.5 ? instance.probability() : 1 - instance.probability());
//                    else
//                        System.out.println("error error error");
//                }
//            }
//            Map<Learning.InstanceToLabel<V>, Double> instanceScores = new HashMap<>();
//            for (Map.Entry<PredictedDataInstance<V, Double>, Map<Label, Double>> instanceEntry : probabilitiesMap.entrySet()) {
//                for (Map.Entry<Label, Double> instanceEntryLabel : instanceEntry.getValue().entrySet()) {
//                    double instanceScore = 0.0;
//                    // Setting label to true and propagating
//                    Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getFixedLabels(instanceEntry.getKey().name()));
//                    newFixedLabels.put(instanceEntryLabel.getKey(), true);
//                    learning.getConstraints().propagate(newFixedLabels);
//                    learning.getFixedLabels(instanceEntry.getKey().name()).keySet().forEach(newFixedLabels::remove);
//                    newFixedLabels.remove(instanceEntryLabel.getKey());
//                    for (Map.Entry<Label, Boolean> labelEntry : newFixedLabels.entrySet()) {
//                        double labelProbability = probabilitiesMap.get(instanceEntry.getKey()).get(labelEntry.getKey());
//                        instanceScore += instanceEntryLabel.getValue() * Math.log(labelEntry.getValue() ? 1 - labelProbability : labelProbability);
//                    }
//                    // Setting label to false and propagating
//                    newFixedLabels = new HashMap<>(learning.getFixedLabels(instanceEntry.getKey().name()));
//                    newFixedLabels.put(instanceEntryLabel.getKey(), false);
//                    learning.getConstraints().propagate(newFixedLabels);
//                    learning.getFixedLabels(instanceEntry.getKey().name()).keySet().forEach(newFixedLabels::remove);
//                    newFixedLabels.remove(instanceEntryLabel.getKey());
//                    for (Map.Entry<Label, Boolean> labelEntry : newFixedLabels.entrySet()) {
//                        double labelProbability = probabilitiesMap.get(instanceEntry.getKey()).get(labelEntry.getKey());
//                        instanceScore -= (1 - instanceEntryLabel.getValue()) * Math.log(labelEntry.getValue() ? 1 - labelProbability : labelProbability);
//                    }
//                    instanceScores.put(new Learning.InstanceToLabel<>(instanceEntryLabel.getKey(), instanceEntry.getKey()), instanceScore);
//                }
//            }
            Map<Learning.InstanceToLabel<V>, Double> instanceScores = new HashMap<>();
            for (Map.Entry<Label, DataSet<PredictedDataInstance<V, Double>>> instanceEntry : dataSets.entrySet()) {
                DataSet<PredictedDataInstance<V, Double>> currentDataSet = instanceEntry.getValue();
                for (PredictedDataInstance<V, Double> instance : currentDataSet)
                    instanceScores.put(new Learning.InstanceToLabel<>(instanceEntry.getKey(), instance),
                                       instance.label() >= 0.5 ? instance.probability() : 1 - instance.probability());
            }
            return instanceScores;
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
