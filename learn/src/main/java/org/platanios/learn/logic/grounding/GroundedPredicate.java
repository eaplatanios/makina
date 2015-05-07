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
    private final R value;

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

    public R getValue() {
        return value;
    }
}
