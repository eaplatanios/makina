package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.formula.Predicate;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GroundPredicate<R> {
    private final long id;
    private final Predicate predicate;
    private final List<Long> predicateArgumentsAssignment;

    private R value;

    public GroundPredicate(long id,
                           Predicate predicate,
                           List<Long> predicateArgumentsAssignment) {
        this.id = id;
        this.predicate = predicate;
        this.predicateArgumentsAssignment = predicateArgumentsAssignment;
        this.value = null;
    }

    public GroundPredicate(long id,
                           Predicate predicate,
                           List<Long> predicateArgumentsAssignment,
                           R value) {
        this.id = id;
        this.predicate = predicate;
        this.predicateArgumentsAssignment = predicateArgumentsAssignment;
        this.value = value;
    }

    public long getId() {
        return id;
    }

    public Predicate getPredicate() {
        return predicate;
    }

    public List<Long> getPredicateArgumentsAssignment() {
        return predicateArgumentsAssignment;
    }

    public void setValue(R value) {
        this.value = value;
    }

    public R getValue() {
        return value;
    }

    @Override
    @SuppressWarnings("unchecked")
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        GroundPredicate<R> that = (GroundPredicate<R>) other;

        if (id != that.id)
            return false;
        if (!predicate.equals(that.predicate))
            return false;
        if (!predicateArgumentsAssignment.equals(that.predicateArgumentsAssignment))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + predicate.hashCode();
        result = 31 * result + predicateArgumentsAssignment.hashCode();
        return result;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        if (predicate.getName() != null)
            stringBuilder.append(predicate.getName()).append("(");
        else
            stringBuilder.append(predicate.getId()).append("(");
        for (int argumentIndex = 0; argumentIndex < predicateArgumentsAssignment.size(); argumentIndex++) {
            stringBuilder.append(predicateArgumentsAssignment.get(argumentIndex).toString());
            if (argumentIndex != predicateArgumentsAssignment.size() - 1)
                stringBuilder.append(", ");
        }
        return stringBuilder.append(")").toString();
    }
}
