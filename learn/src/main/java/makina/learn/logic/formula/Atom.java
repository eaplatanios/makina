package makina.learn.logic.formula;

import makina.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Atom extends Formula {
    private Predicate predicate;
    private List<? extends Term> predicateArguments;

    public Atom(Predicate predicate, List<? extends Term> predicateArguments) {
        this.predicate = predicate;
        this.predicateArguments = predicateArguments;
        if (!predicate.isValidArgumentAssignment(predicateArguments))
            throw new IllegalArgumentException("The types of the provided arguments do not match the types that the " +
                                                       "provided predicate requires.");
    }

    public Predicate getPredicate() {
        return predicate;
    }

    public List<? extends Term> getPredicateArguments() {
        return predicateArguments;
    }

    @Override
    public Set<Variable> getVariables() {
        return predicateArguments.stream()
                .filter(term -> term instanceof Variable)
                .map(term -> (Variable) term)
                .collect(Collectors.toSet());
    }

    @Override
    public List<Variable> getOrderedVariables() {
        return predicateArguments.stream()
                .filter(term -> term instanceof Variable)
                .map(term -> (Variable) term)
                .collect(Collectors.toList());
    }

    @Override
    public Double evaluate(LogicManager logicManager, Map<Long, Long> variableAssignments) {
        List<Long> predicateArgumentValues = new ArrayList<>();
        for (Term predicateArgument : predicateArguments) {
            Long variableAssignment = variableAssignments.getOrDefault(predicateArgument.getId(), null);
            if (variableAssignment != null)
                predicateArgumentValues.add(variableAssignment);
            else
                throw new IllegalArgumentException("The provided variable assignments map does not include " +
                                                           "assignments for all arguments of this predicate.");
        }
        return logicManager.getPredicateAssignmentTruthValue(predicate, predicateArgumentValues);
    }

    @Override
    public Formula toDisjunctiveNormalForm() {
        return this;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder(predicate.toStringWithoutArgumentTypes());
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
