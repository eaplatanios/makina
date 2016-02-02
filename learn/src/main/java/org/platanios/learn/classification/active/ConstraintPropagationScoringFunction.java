package org.platanios.learn.classification.active;

import com.google.common.base.Objects;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.ConstraintSet;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * // TODO: Add builder class.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ConstraintPropagationScoringFunction extends ScoringFunction {
    private final SurpriseFunction surpriseFunction;
    private final boolean useMutualExclusionSpecialCase;
    private final boolean includeFixedVariableTermInTotalSurprise;

    public ConstraintPropagationScoringFunction() {
        this(SurpriseFunction.NEGATIVE_LOGARITHM, false, true);
    }

    public ConstraintPropagationScoringFunction(SurpriseFunction surpriseFunction) {
        this(surpriseFunction, false, true);
    }

    public ConstraintPropagationScoringFunction(boolean useMutualExclusionSpecialCase) {
        this(SurpriseFunction.NEGATIVE_LOGARITHM, useMutualExclusionSpecialCase, true);
    }

    public ConstraintPropagationScoringFunction(SurpriseFunction surpriseFunction,
                                                boolean includeFixedVariableTermInTotalSurprise) {
        this(surpriseFunction, false, includeFixedVariableTermInTotalSurprise);
    }

    public ConstraintPropagationScoringFunction(SurpriseFunction surpriseFunction,
                                                boolean useMutualExclusionSpecialCase,
                                                boolean includeFixedVariableTermInTotalSurprise) {
        this.surpriseFunction = surpriseFunction;
        this.useMutualExclusionSpecialCase = useMutualExclusionSpecialCase;
        this.includeFixedVariableTermInTotalSurprise = includeFixedVariableTermInTotalSurprise;
    }

    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        if (!(learning instanceof ConstrainedLearning))
            throw new IllegalArgumentException("This active learning method can only " +
                                                       "be used with the constrained learner.");
        ConstrainedLearning constrainedLearning = (ConstrainedLearning) learning;
        if (useMutualExclusionSpecialCase && constrainedLearning.getConstraintSet() instanceof ConstraintSet) {
            Set<Constraint> constraints = ((ConstraintSet) constrainedLearning.getConstraintSet()).getConstraints();
            if (constraints.size() == 1 && constraints.iterator().next() instanceof MutualExclusionConstraint)
                return instanceToLabel.getProbability();
        }
        if (useMutualExclusionSpecialCase)
            return instanceToLabel.getProbability();
        double score = 0.0;
        // Setting label to true and propagating
        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instanceToLabel.getInstance()));
        fixedLabels.put(instanceToLabel.getLabel(), true);
        constrainedLearning.getConstraintSet().propagate(fixedLabels);
        learning.getLabels(instanceToLabel.getInstance()).keySet().forEach(fixedLabels::remove);
        if (!includeFixedVariableTermInTotalSurprise)
            fixedLabels.remove(instanceToLabel.getLabel());
        double scoreTerm = 0.0;
        for (Map.Entry<Label, Boolean> labelEntry : fixedLabels.entrySet()) {
            double labelProbability = learning.new InstanceToLabel(instanceToLabel.getInstance(),
                                                                   labelEntry.getKey()).getProbability();
            scoreTerm += surpriseFunction.surprise(labelEntry.getValue() ? labelProbability : 1 - labelProbability);
        }
        score += instanceToLabel.getProbability() * scoreTerm;
        // Setting label to false and propagating
        fixedLabels = new HashMap<>(learning.getLabels(instanceToLabel.getInstance()));
        fixedLabels.put(instanceToLabel.getLabel(), false);
        constrainedLearning.getConstraintSet().propagate(fixedLabels);
        learning.getLabels(instanceToLabel.getInstance()).keySet().forEach(fixedLabels::remove);
        if (!includeFixedVariableTermInTotalSurprise)
            fixedLabels.remove(instanceToLabel.getLabel());
        scoreTerm = 0.0;
        for (Map.Entry<Label, Boolean> labelEntry : fixedLabels.entrySet()) {
            double labelProbability = learning.new InstanceToLabel(instanceToLabel.getInstance(),
                                                                   labelEntry.getKey()).getProbability();
            scoreTerm += surpriseFunction.surprise(labelEntry.getValue() ? labelProbability : 1 - labelProbability);
        }
        score += (1 - instanceToLabel.getProbability()) * scoreTerm;
        return score;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        ConstraintPropagationScoringFunction that = (ConstraintPropagationScoringFunction) other;

        return Objects.equal(surpriseFunction, that.surpriseFunction)
                && Objects.equal(useMutualExclusionSpecialCase, that.useMutualExclusionSpecialCase)
                && Objects.equal(includeFixedVariableTermInTotalSurprise, that.includeFixedVariableTermInTotalSurprise);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass(),
                                surpriseFunction,
                                useMutualExclusionSpecialCase,
                                includeFixedVariableTermInTotalSurprise);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        if (!useMutualExclusionSpecialCase) {
            switch (surpriseFunction) {
                case NEGATIVE_LOGARITHM:
                    stringBuilder.append("LOG");
                    break;
                case ONE_MINUS_PROBABILITY:
                    stringBuilder.append("LINEAR");
                    break;
            }
            if (!includeFixedVariableTermInTotalSurprise)
                stringBuilder.append("-EXCL");
            return stringBuilder.append("-CP").toString();
        } else {
            stringBuilder.append("PROBABILITY");
            return stringBuilder.append("-CP").toString();
        }
    }
}
