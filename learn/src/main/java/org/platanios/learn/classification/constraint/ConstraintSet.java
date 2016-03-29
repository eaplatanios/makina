package org.platanios.learn.classification.constraint;

import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.active.ConstrainedLearning;
import org.platanios.learn.data.DataInstance;
import org.platanios.math.matrix.Vector;

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

    public Set<Constraint> getConstraints() {
        return constraints;
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

    @Override
    public int propagate(Map<Label, Boolean> fixedLabels, ConstrainedLearning learning, DataInstance<Vector> instance) {
        int previousNumberOfFixedLabels;
        int currentNumberOfFixedLabels = 0;
        do {
            previousNumberOfFixedLabels = currentNumberOfFixedLabels;
            for (Constraint constraint : constraints)
                currentNumberOfFixedLabels += constraint.propagate(fixedLabels, learning, instance);
        } while (currentNumberOfFixedLabels != previousNumberOfFixedLabels);
        return currentNumberOfFixedLabels;
    }
}
