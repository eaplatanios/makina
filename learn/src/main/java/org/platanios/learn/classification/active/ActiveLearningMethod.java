package org.platanios.learn.classification.active;

import com.google.common.collect.Iterables;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.ConstraintSet;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum ActiveLearningMethod {
    RANDOM {
        @Override
        public Double computeInformationGainHeuristicValue(Learning learning,
                                                           Learning.InstanceToLabel instanceToLabel) {
            return new Random().nextDouble();
        }
    },
    UNCERTAINTY_HEURISTIC {
        @Override
        public Double computeInformationGainHeuristicValue(Learning learning,
                                                           Learning.InstanceToLabel instanceToLabel) {
            return entropy(instanceToLabel.getProbability());
        }
    },
    CONSTRAINT_PROPAGATION_HEURISTIC {
        @Override
        public Double computeInformationGainHeuristicValue(Learning learning,
                                                           Learning.InstanceToLabel instanceToLabel) {
            if (!(learning instanceof ConstrainedLearning))
                throw new IllegalArgumentException("This active learning method can only " +
                                                           "be used with the constrained learner.");

            if (((ConstrainedLearning) learning).getConstraintSet() instanceof ConstraintSet
                    && ((ConstraintSet) ((ConstrainedLearning) learning).getConstraintSet()).getConstraints().size() == 1
                    && Iterables.getOnlyElement(((ConstraintSet) ((ConstrainedLearning) learning).getConstraintSet()).getConstraints()) instanceof MutualExclusionConstraint) {
                return instanceToLabel.getProbability();
            }
            List<Double> instanceScoreTerms = new ArrayList<>();
            // Setting label to true and propagating
            Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instanceToLabel.getInstance()));
            fixedLabels.put(instanceToLabel.getLabel(), true);
            ((ConstrainedLearning) learning).getConstraintSet().propagate(fixedLabels);
            learning.getLabels(instanceToLabel.getInstance()).keySet().forEach(fixedLabels::remove);
            fixedLabels.remove(instanceToLabel.getLabel());
//                    instanceScoreTerms.add(Math.log(instanceEntryLabel.getValue()));
//                    instanceScoreTerms.add(instanceEntryLabel.getValue());
            double term = 0.0;
            for (Map.Entry<Label, Boolean> labelEntry : fixedLabels.entrySet()) {
                double labelProbability = learning.new InstanceToLabel(instanceToLabel.getInstance(), labelEntry.getKey()
                ).getProbability();
                term += instanceToLabel.getProbability() *
                        (labelEntry.getValue() ? 1 - labelProbability : labelProbability);
            }
            instanceScoreTerms.add(term);
//                    if (term > 0.0)
//                        instanceScoreTerms.add(Math.log(instanceEntryLabel.getValue()) + Math.log(term));
            // Setting label to false and propagating
            fixedLabels = new HashMap<>(learning.getLabels(instanceToLabel.getInstance()));
            fixedLabels.put(instanceToLabel.getLabel(), false);
            ((ConstrainedLearning) learning).getConstraintSet().propagate(fixedLabels);
            learning.getLabels(instanceToLabel.getInstance()).keySet().forEach(fixedLabels::remove);
            fixedLabels.remove(instanceToLabel.getLabel());
            term = 0.0;
            for (Map.Entry<Label, Boolean> labelEntry : fixedLabels.entrySet()) {
                double labelProbability = learning.new InstanceToLabel(instanceToLabel.getInstance(), labelEntry.getKey()
                ).getProbability();
                term += (1 - instanceToLabel.getProbability()) *
                        (labelEntry.getValue() ? 1 - labelProbability : labelProbability);
            }
//                    if (term > 0.0)
//                        instanceScoreTerms.add(Math.log(1 - instanceEntryLabel.getValue()) + Math.log(term));
//                    if (instanceScoreTerms.size() > 0)
            return instanceScoreTerms.stream().mapToDouble(number -> number).sum();
//                    else
//                        instanceScores.put(
//                                new Learning.InstanceToLabel(instanceEntryLabel.getKey(), instanceEntry.getKey()),
//                                Double.NEGATIVE_INFINITY
//                        );
//                    instanceScores.put(
//                            new Learning.InstanceToLabel(instanceEntryLabel.getKey(), instanceEntry.getKey()),
//                            Math.log(instanceEntryLabel.getValue())
//                    );
        }
    };

    public abstract Double computeInformationGainHeuristicValue(Learning learning,
                                                                Learning.InstanceToLabel instanceToLabel);

    protected double entropy(double probability) {
        if (probability > 0)
            return -probability * Math.log(probability) - (1 - probability) * Math.log(1 - probability);
        else
            return 0;
    }
}
