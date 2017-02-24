package makina.learn.classification.active;

import com.google.common.base.Objects;
import makina.learn.classification.Label;
import makina.learn.classification.constraint.Constraint;
import makina.learn.classification.constraint.ConstraintSet;
import makina.learn.classification.constraint.MutualExclusionConstraint;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * // TODO: Add builder class.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ExpectedLabelsFixedScoringFunction extends ScoringFunction {
    private final boolean includeFixedVariableTermInTotalSurprise;

    public ExpectedLabelsFixedScoringFunction() {
        this(true);
    }

    public ExpectedLabelsFixedScoringFunction(boolean includeFixedVariableTermInTotalSurprise) {
        this.includeFixedVariableTermInTotalSurprise = includeFixedVariableTermInTotalSurprise;
    }

    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        if (!(learning instanceof ConstrainedLearning))
            throw new IllegalArgumentException("This active learning method can only " +
                    "be used with the constrained learner.");
        ConstrainedLearning constrainedLearning = (ConstrainedLearning) learning;
        double score = 0.0;
        // Setting label to true and propagating
        Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instanceToLabel.getInstance()));
        fixedLabels.put(instanceToLabel.getLabel(), true);
        constrainedLearning.getConstraintSet().propagate(fixedLabels);
        learning.getLabels(instanceToLabel.getInstance()).keySet().forEach(fixedLabels::remove);
        if (!includeFixedVariableTermInTotalSurprise)
            fixedLabels.remove(instanceToLabel.getLabel());
        score += instanceToLabel.getProbability() * fixedLabels.size();
        // Setting label to false and propagating
        fixedLabels = new HashMap<>(learning.getLabels(instanceToLabel.getInstance()));
        fixedLabels.put(instanceToLabel.getLabel(), false);
        constrainedLearning.getConstraintSet().propagate(fixedLabels);
        learning.getLabels(instanceToLabel.getInstance()).keySet().forEach(fixedLabels::remove);
        if (!includeFixedVariableTermInTotalSurprise)
            fixedLabels.remove(instanceToLabel.getLabel());
        score += (1 - instanceToLabel.getProbability()) * fixedLabels.size();
        return score;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        ExpectedLabelsFixedScoringFunction that = (ExpectedLabelsFixedScoringFunction) other;

        return Objects.equal(includeFixedVariableTermInTotalSurprise, that.includeFixedVariableTermInTotalSurprise);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass(), includeFixedVariableTermInTotalSurprise);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("EL").toString();
        if (!includeFixedVariableTermInTotalSurprise)
            stringBuilder.append("-EXCL");
        return stringBuilder.toString();
    }
}
