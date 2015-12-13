package org.platanios.learn.classification.active;

import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConstraintSet implements Constraint {
    private final Set<Constraint> constraints;

    public ConstraintSet(Set<Constraint> constraints) {
        this.constraints = constraints;
    }

    @Override
    public int propagate(Map<Label, Boolean> fixedLabels) {
        int previousNumberOfFixedLabels;
        int currentNumberOfFixedLabels = 0;
        do {
            previousNumberOfFixedLabels = currentNumberOfFixedLabels;
            for (Constraint constraint : constraints)
                currentNumberOfFixedLabels += constraint.propagate(fixedLabels);
        } while (currentNumberOfFixedLabels != previousNumberOfFixedLabels);
        return currentNumberOfFixedLabels;
    }
}
