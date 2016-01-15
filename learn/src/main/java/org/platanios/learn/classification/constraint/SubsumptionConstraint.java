package org.platanios.learn.classification.constraint;

import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.active.ConstrainedLearning;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SubsumptionConstraint implements Constraint {
    private final Label parentLabel;
    private final Label childLabel;

    public SubsumptionConstraint(Label parentLabel, Label childLabel) {
        this.parentLabel = parentLabel;
        this.childLabel = childLabel;
    }

    public Label getParentLabel() {
        return parentLabel;
    }

    public Label getChildLabel() {
        return childLabel;
    }

    @Override
    public int propagate(Map<Label, Boolean> fixedLabels) {
        int numberOfLabelsFixed = 0;
        if (!fixedLabels.getOrDefault(parentLabel, true)) {
            if (!fixedLabels.containsKey(childLabel)) {
                fixedLabels.put(childLabel, false);
                numberOfLabelsFixed++;
            } else if (fixedLabels.get(childLabel)) {
                throw new IllegalStateException("The provided set of constraints is unsatisfiable.");
            }
        } else if (fixedLabels.getOrDefault(childLabel, false)) {
            if (!fixedLabels.containsKey(parentLabel)) {
                fixedLabels.put(parentLabel, true);
                numberOfLabelsFixed++;
            } else if (!fixedLabels.get(parentLabel)) {
                throw new IllegalStateException("The provided set of constraints is unsatisfiable.");
            }
        }
        return numberOfLabelsFixed;
    }

    @Override
    public int propagate(Map<Label, Boolean> fixedLabels, ConstrainedLearning learning, DataInstance<Vector> instance) {
        int numberOfLabelsFixed = 0;
        if (!fixedLabels.getOrDefault(parentLabel, true)) {
            if (!fixedLabels.containsKey(childLabel)) {
                learning.labelInstanceWithoutPropagation(instance, childLabel, false);
                numberOfLabelsFixed++;
            } else if (fixedLabels.get(childLabel)) {
                throw new IllegalStateException("The provided set of constraints is unsatisfiable.");
            }
        } else if (fixedLabels.getOrDefault(childLabel, false)) {
            if (!fixedLabels.containsKey(parentLabel)) {
                learning.labelInstanceWithoutPropagation(instance, parentLabel, true);
                numberOfLabelsFixed++;
            } else if (!fixedLabels.get(parentLabel)) {
                throw new IllegalStateException("The provided set of constraints is unsatisfiable.");
            }
        }
        return numberOfLabelsFixed;
    }
}
