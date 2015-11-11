package org.platanios.learn.logic;

import org.platanios.learn.logic.formula.EntityType;
import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.formula.Variable;
import org.platanios.learn.logic.grounding.GroundPredicate;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class InMemoryLogicManager implements LogicManager {
    private final Logic logic;

    private final Map<Long, EntityType> entityTypes = new HashMap<>();
    private final Map<Long, Predicate> predicates = new HashMap<>();
    private final Map<Long, Map<List<Long>, GroundPredicate>> groundedPredicates = new HashMap<>();
    private final Map<Long, GroundPredicate> groundedPredicatesMap = new HashMap<>();
    private final List<Long> closedPredicateIdentifiers = new ArrayList<>();

    private long newEntityTypeIdentifier = 0;
    private long newPredicateIdentifier = 0;
    private long newPredicateGroundingIdentifier = 0;

    public InMemoryLogicManager(Logic logic) {
        this.logic = logic;
    }

    public Logic logic() {
        return logic;
    }

    public EntityType addEntityType(Set<Long> allowedValues) {
        return addEntityType(null, allowedValues);
    }

    public EntityType addEntityType(String name, Set<Long> allowedValues) {
        EntityType entityType = new EntityType(newEntityTypeIdentifier, name, allowedValues);
        entityTypes.put(newEntityTypeIdentifier++, entityType);
        return entityType;
    }

    public Predicate addPredicate(List<EntityType> argumentTypes, boolean closed) {
        return addPredicate(null, argumentTypes, closed);
    }

    public Predicate addPredicate(String name, List<EntityType> argumentTypes, boolean closed) {
        Predicate predicate = new Predicate(newPredicateIdentifier, name, argumentTypes);
        predicates.put(newPredicateIdentifier, predicate);
        groundedPredicates.put(newPredicateIdentifier, new HashMap<>());
        if (closed)
            closedPredicateIdentifiers.add(newPredicateIdentifier);
        newPredicateIdentifier++;
        return predicate;
    }

    public GroundPredicate addGroundPredicate(Predicate predicate, List<Long> argumentAssignments) {
        return addGroundPredicate(predicate, argumentAssignments, null);
    }

    public GroundPredicate addGroundPredicate(Predicate predicate, List<Long> argumentAssignments, Double value) {
        if (!groundedPredicates.containsKey(predicate.getId()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");
        if (groundedPredicates.get(predicate.getId()).containsKey(argumentAssignments)) {
            GroundPredicate groundPredicate =
                    groundedPredicates.get(predicate.getId()).get(argumentAssignments);
            if (!(groundPredicate.getValue() == null && value == null) && !groundPredicate.getValue().equals(value))
                throw new IllegalArgumentException("A grounding for the predicate corresponding to the provided " +
                                                           "identifier and for the provided argument assignments has " +
                                                           "already been added to this logic manager with a " +
                                                           "different value.");
            else {
                return groundPredicate;
            }
        } else {
            GroundPredicate groundPredicate = new GroundPredicate(newPredicateGroundingIdentifier++,
                                                                  predicate,
                                                                  argumentAssignments,
                                                                  value);
            groundedPredicatesMap.put(groundPredicate.getId(), groundPredicate);
            groundedPredicates.get(predicate.getId()).put(argumentAssignments, groundPredicate);
            return groundPredicate;
        }
    }

    public boolean checkIfGroundPredicateExists(Predicate predicate, List<Long> argumentAssignments) {
        if (!groundedPredicates.containsKey(predicate.getId()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");

        return groundedPredicates.get(predicate.getId()).containsKey(argumentAssignments)
                || closedPredicateIdentifiers.contains(predicate.getId());
    }

    // TODO: Maybe change return type to long?
    public long getNumberOfGroundPredicates() {
        return groundedPredicatesMap.size();
    }

    public List<GroundPredicate> getGroundPredicates() {
        List<GroundPredicate> groundPredicates = new ArrayList<>();
        for (Map<List<Long>, GroundPredicate> groundedPredicatesSet : this.groundedPredicates.values())
            groundPredicates.addAll(groundedPredicatesSet.values().stream().collect(Collectors.toList()));
        return groundPredicates;
    }

    // TODO: Fix the way in which the grounded predicates are stored in this manager.
    public GroundPredicate getGroundPredicate(long identifier) {
        return groundedPredicatesMap.get(identifier);
    }

    public GroundPredicate getGroundPredicate(Predicate predicate, List<Long> argumentAssignments) {
        if (!groundedPredicates.containsKey(predicate.getId()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");
        if (!groundedPredicates.get(predicate.getId()).containsKey(argumentAssignments))
            throw new IllegalArgumentException("A grounding for the predicate corresponding to the provided " +
                                                       "identifier and for the provided argument assignments has " +
                                                       "not been added to this logic manager.");

        return groundedPredicates.get(predicate.getId()).get(argumentAssignments);
    }

    // TODO: Maybe change return type to long?
    public long getNumberOfEntityTypes() {
        return entityTypes.keySet().size();
    }

    public EntityType getEntityType(long identifier) {
        return entityTypes.get(identifier);
    }

    /**
     * TODO: Improve this implementation.
     *
     * @param name
     * @return
     */
    public EntityType getEntityType(String name) {
        for (Map.Entry<Long, EntityType> variableEntry : entityTypes.entrySet())
            if (variableEntry.getValue().getName().equals(name))
                return variableEntry.getValue();
        return null;
    }

    public Set<Long> getVariableValues(Variable variable) {
        return entityTypes.get(variable.getType().getId()).getPrimitiveAllowedValues();
    }

    public Predicate getPredicate(long identifier) {
        return predicates.get(identifier);
    }

    /**
     * TODO: Improve this implementation.
     *
     * @param name
     * @return
     */
    public Predicate getPredicate(String name) {
        for (Map.Entry<Long, Predicate> predicateEntry : predicates.entrySet())
            if (predicateEntry.getValue().getName().equals(name))
                return predicateEntry.getValue();
        return null;
    }

    public List<Predicate> getClosedPredicates() {
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
    public Double getPredicateAssignmentTruthValue(Predicate predicate, List<Long> argumentAssignments) {
        if (!groundedPredicates.containsKey(predicate.getId()))
            throw new IllegalArgumentException("The provided predicate identifier does not match any of the " +
                                                       "predicates currently stored in this logic manager.");
        if (!groundedPredicates.get(predicate.getId()).containsKey(argumentAssignments))
            if (closedPredicateIdentifiers.contains(predicate.getId()))
                return logic.falseValue();
            else
                return null;
        return groundedPredicates.get(predicate.getId()).get(argumentAssignments).getValue();
    }
}
