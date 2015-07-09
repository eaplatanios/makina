package org.platanios.learn.logic;

import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.formula.*;
import org.platanios.learn.logic.grounding.GroundPredicate;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DatabaseLogicManager implements LogicManager {
    private final Logic logic;
    private final DatabaseManager databaseManager;

//    private final Map<Long, Set<Long>> entityTypeAllowedValues = new HashMap<>();

    public DatabaseLogicManager(Logic logic) {
        this.logic = logic;
        this.databaseManager = new DatabaseManager.Builder().build();
    }

    public DatabaseLogicManager(Logic logic, DatabaseManager databaseManager) {
        this.logic = logic;
        this.databaseManager = databaseManager;
    }

    public Logic logic() {
        return logic;
    }

    public EntityType addEntityType(Set<Long> allowedValues) {
        return addEntityType(null, allowedValues);
    }

    public EntityType addEntityType(String name, Set<Long> allowedValues) {
        return databaseManager.addEntityType(name, allowedValues);
    }

    public Predicate addPredicate(List<EntityType> argumentTypes, boolean closed) {
        return addPredicate(null, argumentTypes, closed);
    }

    public Predicate addPredicate(String name, List<EntityType> argumentTypes, boolean closed) {
        return databaseManager.addPredicate(name, argumentTypes, closed);
    }

    public GroundPredicate addGroundPredicate(Predicate predicate, List<Long> argumentAssignments) {
        return addGroundPredicate(predicate, argumentAssignments, null);
    }

    public GroundPredicate addGroundPredicate(Predicate predicate, List<Long> argumentAssignments, Double value) {
        return databaseManager.addGroundPredicate(
                predicate.getId(),
                argumentAssignments,
                value
        );
    }

    public GroundPredicate addGroundPredicate(GroundPredicate groundPredicate) {
        return databaseManager.addGroundPredicate(groundPredicate);
    }

    public List<GroundPredicate> addGroundPredicates(List<GroundPredicate> groundPredicates) {
        return databaseManager.addGroundPredicates(groundPredicates);
    }

    public boolean checkIfGroundPredicateExists(Predicate predicate, List<Long> argumentAssignments) {
        return databaseManager.checkIfGroundPredicateExists(predicate.getId(), argumentAssignments);
    }

    public long getNumberOfGroundPredicates() {
        return databaseManager.getNumberOfGroundPredicates();
    }

    public List<GroundPredicate> getGroundPredicates() {
        return databaseManager.getGroundPredicates();
    }

    public GroundPredicate getGroundPredicate(long identifier) {
        return databaseManager.getGroundPredicate(identifier);
    }

    public GroundPredicate getGroundPredicate(Predicate predicate, List<Long> variableAssignments) {
        return databaseManager.getGroundPredicate(predicate.getId(), variableAssignments);
    }

    public long getNumberOfEntityTypes() {
        return databaseManager.getNumberOfEntityTypes();
    }

    public EntityType getEntityType(long identifier) {
        return databaseManager.getEntityType(identifier);
    }

    public EntityType getEntityType(String name) {
        return databaseManager.getEntityType(name);
    }

    public Set<Long> getVariableValues(Variable variable) {
        return databaseManager.getEntityTypeAllowedValues(variable.getType().getId())
                .stream()
                .map(EntityTypeValue::getValue)
                .collect(Collectors.toSet());
    }

    public Predicate getPredicate(long identifier) {
        return databaseManager.getPredicate(identifier);
    }

    public Predicate getPredicate(String name) {
        return databaseManager.getPredicate(name);
    }

    public List<Predicate> getClosedPredicates() {
        return databaseManager.getClosedPredicates();
    }

    /**
     * Note that this method returns null if there is no value stored for the provided arguments assignment.
     *
     * @param predicate
     * @param variablesAssignment
     * @return
     */
    @SuppressWarnings("unchecked")
    public Double getPredicateAssignmentTruthValue(Predicate predicate, List<Long> variablesAssignment) {
        return databaseManager.getPredicateAssignmentTruthValue(predicate, variablesAssignment, logic);
    }

    public DatabaseManager.PartialGroundedFormula getMatchingGroundPredicates(List<Atom> atoms) {
        return databaseManager.getMatchingGroundPredicates(atoms, logic);
    }
}
