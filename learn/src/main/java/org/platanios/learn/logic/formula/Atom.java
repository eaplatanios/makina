package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Atom<T> extends Formula<T> {
    private Predicate<T> predicate;
    private List<? extends Term<T>> predicateArguments;

    public Atom(Predicate<T> predicate, List<? extends Term<T>> predicateArguments) {
        this.predicate = predicate;
        this.predicateArguments = predicateArguments;
        if (!predicate.isValidArgumentAssignment(predicateArguments))
            throw new IllegalArgumentException("The types of the provided arguments do not match the types that the " +
                                                       "provided predicate requires.");
    }

    public Predicate<T> getPredicate() {
        return predicate;
    }

    public List<? extends Term<T>> getPredicateArguments() {
        return predicateArguments;
    }

    @Override
    public Set<Variable<T>> getVariables() {
        return predicateArguments.stream()
                .filter(term -> term instanceof Variable)
                .map(term -> (Variable<T>) term)
                .collect(Collectors.toSet());
    }

    @Override
    public List<Variable<T>> getOrderedVariables() {
        return predicateArguments.stream()
                .filter(term -> term instanceof Variable)
                .map(term -> (Variable<T>) term)
                .collect(Collectors.toList());
    }

    @Override
    public <R> R evaluate(LogicManager<T, R> logicManager, Map<Variable<T>, T> variableAssignments) {
        List<T> predicateArgumentValues = new ArrayList<>();
        for (Term<T> predicateArgument : predicateArguments) {
            T variableAssignment = variableAssignments.getOrDefault(predicateArgument, null);
            if (variableAssignment != null)
                predicateArgumentValues.add(variableAssignment);
            else
                throw new IllegalArgumentException("The provided variable assignments map does not include " +
                                                           "assignments for all arguments of this predicate.");
        }
        return logicManager.getPredicateAssignmentTruthValue(predicate, predicateArgumentValues);
    }

    @Override
    public Formula<T> toDisjunctiveNormalForm() {
        return this;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder(predicate.toStringNoArgumentTypes());
        stringBuilder.append("(");
        for (int argumentIndex = 0; argumentIndex < predicateArguments.size(); argumentIndex++) {
            stringBuilder.append(predicateArguments.get(argumentIndex));
            if (argumentIndex < predicateArguments.size() - 1)
                stringBuilder.append(", ");
            else
                stringBuilder.append(")");

        }
        return stringBuilder.toString();
    }
}
