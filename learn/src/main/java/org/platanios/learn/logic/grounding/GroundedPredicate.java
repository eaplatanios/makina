package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.formula.Predicate;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GroundedPredicate<T, R> {
    private final long identifier;
    private final Predicate<T> predicate;
    private final List<T> predicateArgumentsAssignment;

    private R value;

    public GroundedPredicate(long identifier,
                             Predicate<T> predicate,
                             List<T> predicateArgumentsAssignment) {
        this.identifier = identifier;
        this.predicate = predicate;
        this.predicateArgumentsAssignment = predicateArgumentsAssignment;
        this.value = null;
    }

    public GroundedPredicate(long identifier,
                             Predicate<T> predicate,
                             List<T> predicateArgumentsAssignment,
                             R value) {
        this.identifier = identifier;
        this.predicate = predicate;
        this.predicateArgumentsAssignment = predicateArgumentsAssignment;
        this.value = value;
    }

    public long getIdentifier() {
        return identifier;
    }

    public Predicate<T> getPredicate() {
        return predicate;
    }

    public List<T> getPredicateArgumentsAssignment() {
        return predicateArgumentsAssignment;
    }

    public void setValue(R value) {
        this.value = value;
    }

    public R getValue() {
        return value;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        if (predicate.getName() != null)
            stringBuilder.append(predicate.getName()).append("(");
        else
            stringBuilder.append(predicate.getIdentifier()).append("(");
        for (int argumentIndex = 0; argumentIndex < predicateArgumentsAssignment.size(); argumentIndex++) {
            stringBuilder.append(predicateArgumentsAssignment.get(argumentIndex).toString());
            if (argumentIndex != predicateArgumentsAssignment.size() - 1)
                stringBuilder.append(", ");
        }
        return stringBuilder.append(")").toString();
    }
}
