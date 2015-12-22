package org.platanios.learn.classification.constraint;

import org.platanios.learn.classification.Label;

import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MutualExclusionConstraint implements Constraint {
    private final Set<Label> labels;

    public MutualExclusionConstraint(Set<Label> labels) {
        this.labels = labels;
    }

    public Set<Label> getLabels() {
        return labels;
    }

    @Override
    public int propagate(Map<Label, Boolean> fixedLabels) {
        int numberOfLabelsFixed = 0;
        for (Map.Entry<Label, Boolean> labelEntry : fixedLabels.entrySet())
            if (labels.contains(labelEntry.getKey()) && labelEntry.getValue()) {
                for (Label label : labels)
                    if (!label.equals(labelEntry.getKey())) {
                        if (!fixedLabels.containsKey(label)) {
                            fixedLabels.put(label, false);
                            numberOfLabelsFixed++;
                        } else if (fixedLabels.get(label)) {
                            throw new IllegalStateException("The provided set of constraints is unsatisfiable.");
                        }
                    }
                break;
            }
        return numberOfLabelsFixed;
    }
}
