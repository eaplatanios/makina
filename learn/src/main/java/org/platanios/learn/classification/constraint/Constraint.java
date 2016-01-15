package org.platanios.learn.classification.constraint;

import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.active.ConstrainedLearning;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.Map;

/**
 * // TODO: Add equals(), hashCode(), and toString() implementations.
 *
 * @author Emmanouil Antonios Platanios
 */
public interface Constraint {
    /**
     * Propagates the constraints when fixing the provided labels to the provided values and returns the number of other
     * labels fixed by that constraint propagation, while also adding the newly fixed labels with their corresponding
     * values to the provided map and also while ignoring all the previously fixed labels provided in the process. By
     * ignoring here we mean that the propagation of constraints stops at those labels, since their values are assumed
     * to have already been fixed earlier in the constraint propagation process.
     *
     * TODO: Check visited constraints instead of fixed labels for efficiency.
     *
     * @param   fixedLabels Map containing the labels whose values have been fixed, along with those corresponding
     *                      values.
     *
     * @return              The number of other labels fixed by propagating the constraints.
     */
    int propagate(Map<Label, Boolean> fixedLabels);

    int propagate(Map<Label, Boolean> fixedLabels, ConstrainedLearning learning, DataInstance<Vector> instance);
}
