package org.platanios.learn.logic;

import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.formula.Variable;
import org.platanios.learn.logic.formula.VariableType;
import org.platanios.learn.logic.grounding.GroundedPredicate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogicManager<T, R> {
    private final Logic<R> logic;

    private final Map<Long, VariableType<T>> variableTypes = new HashMap<>();
    private final Map<Long, Variable<T>> variables = new HashMap<>();
    private final Map<Long, List<T>> variableValues = new HashMap<>();
    private final Map<Long, Predicate<T>> predicates = new HashMap<>();
    private final Map<Long, Map<List<T>, GroundedPredicate<T, R>>> predicateGroundings = new HashMap<>();
    private final List<Long> closedPredicateIdentifiers = new ArrayList<>();

    private long newVariableTypeIdentifier = 0;
    private long newVariableIdentifier = 0;
    private long newPredicateIdentifier = 0;
    private long newPredicateGroundingIdentifier = 0;

    public LogicManager(Logic<R> logic) {
        this.logic = logic;
    }

    public Logic<R> logic() {
        return logic;
    }

    public VariableType<T> addVariableType(Class<T> valueType) {
        return addVariableType(null, valueType);
    }

    public VariableType<T> addVariableType(String name, Class<T> valueType) {
        VariableType<T> variableType = new VariableType<>(newVariableTypeIdentifier, valueType).setName(name);
        variableTypes.put(newVariableTypeIdentifier++, variableType);
        return variableType;
    }

    public Variable<T> addVariable(List<T> allowedValues, VariableType<T> type) {
        return addVariable(null, allowedValues, type);
    }

    public Variable<T> addVariable(String name, List<T> allowedValues, VariableType<T> type) {
        if (!variableTypes.containsKey(type.getIdentifier()))
            throw new IllegalArgumentException("The provided variable type identifier does not match any of the " +
                                                       "variable types currently stored in this logic manager.");
        if (!variableTypes.get(type.getIdentifier()).equals(type))
            throw new IllegalArgumentException("The provided variable type identifier does not match the type with " +
                                                       "the same identifier currently stored in this logic manager.");
        Variable<T> variable = new Variable<>(newVariableIdentifier, type).setName(name);
        variableValues.put(newVariableIdentifier, allowedValues);
        variables.put(newVariableIdentifier++, variable);
        return variable;
    }

    public Predicate<T> addPredicate(List<VariableType<T>> argumentTypes, boolean closed) {
        return addPredicate(null, argumentTypes, closed);
    }

    public Predicate<T> addPredicate(String name, List<VariableType<T>> argumentTypes, boolean closed) {
        Predicate<T> predicate = new Predicate<>(newPredicateIdentifier, argumentTypes).setName(name);
        predicates.put(newPredicateIdentifier, predicate);
        predicateGroundings.put(newPredicateIdentifier, new HashMap<>());
        if (closed)
            closedPredicateIdentifiers.add(newPredicateIdentifier);
        newPredicateIdentifier++;
        return predicate;
    }

    public GroundedPredicate<T, R> addGroundedPredicate(Predicate<T> predicate, List<T> argumentAssignments) {
        return addGroundedPredicate(predicate, argumentAssignments, null);
    }

    public GroundedPredicate<T, R> addGroundedPredicate(Predicate<T> predicate, List<T> argumentAssignments, R value) {
        if (!predicateGroundings.containsKey(predicate.getIdentifier()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");
        if (predicateGroundings.get(predicate.getIdentifier()).containsKey(argumentAssignments)) {
            GroundedPredicate<T, R> groundedPredicate =
                    predicateGroundings.get(predicate.getIdentifier()).get(argumentAssignments);
            if (!groundedPredicate.getValue().equals(value))
                throw new IllegalArgumentException("A grounding for the predicate corresponding to the provided " +
                                                           "identifier and for the provided argument assignments has " +
                                                           "already been added to this logic manager with a " +
                                                           "different value.");
            else {
                return groundedPredicate;
            }
        } else {
            GroundedPredicate<T, R> groundedPredicate = new GroundedPredicate<>(newPredicateGroundingIdentifier++,
                                                                                predicate,
                                                                                argumentAssignments,
                                                                                value);
            predicateGroundings.get(predicate.getIdentifier()).put(argumentAssignments, groundedPredicate);
            return groundedPredicate;
        }
    }

    public boolean predicateGroundingExists(Predicate<T> predicate, List<T> argumentAssignments) {
        if (!predicateGroundings.containsKey(predicate.getIdentifier()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");

        return predicateGroundings.get(predicate.getIdentifier()).containsKey(argumentAssignments);
    }

    public GroundedPredicate<T, R> getGroundedPredicate(Predicate<T> predicate, List<T> argumentAssignments) {
        if (!predicateGroundings.containsKey(predicate.getIdentifier()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");
        if (!predicateGroundings.get(predicate.getIdentifier()).containsKey(argumentAssignments))
            throw new IllegalArgumentException("A grounding for the predicate corresponding to the provided " +
                                                       "identifier and for the provided argument assignments has " +
                                                       "not been added to this logic manager.");

        return predicateGroundings.get(predicate.getIdentifier()).get(argumentAssignments);
    }

    public Variable<T> getVariable(long identifier) {
        return variables.get(identifier);
    }

    /**
     * TODO: Improve this implementation.
     *
     * @param name
     * @return
     */
    public Variable<T> getVariable(String name) {
        for (Map.Entry<Long, Variable<T>> variableEntry : variables.entrySet())
            if (variableEntry.getValue().getName().equals(name))
                return variableEntry.getValue();
        return null;
    }

    public List<T> getVariableValues(Variable<T> variable) {
        return variableValues.get(variable.getIdentifier());
    }

    public Predicate<T> getPredicate(long identifier) {
        return predicates.get(identifier);
    }

    /**
     * TODO: Improve this implementation.
     *
     * @param name
     * @return
     */
    public Predicate<T> getPredicate(String name) {
        for (Map.Entry<Long, Predicate<T>> predicateEntry : predicates.entrySet())
            if (predicateEntry.getValue().getName().equals(name))
                return predicateEntry.getValue();
        return null;
    }

    public List<Predicate<T>> getClosedPredicates() {
        return closedPredicateIdentifiers.stream()
                .map(predicates::get)
                .collect(Collectors.toList());
    }

    /**
     * Note that this method returns null if there is no value stored for the provided arguments assignment.
     *
     * @param predicate
     * @param argumentAssignments
     * @return
     */
    public R getPredicateAssignmentTruthValue(Predicate<T> predicate, List<T> argumentAssignments) {
        if (!predicateGroundings.containsKey(predicate.getIdentifier()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");
        if (!predicateGroundings.get(predicate.getIdentifier()).containsKey(argumentAssignments))
            if (closedPredicateIdentifiers.contains(predicate.getIdentifier()))
                return logic.falseValue();
            else
                return null;
        return predicateGroundings.get(predicate.getIdentifier()).get(argumentAssignments).getValue();
    }
}
