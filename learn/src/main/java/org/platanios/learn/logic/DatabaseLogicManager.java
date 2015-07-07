package org.platanios.learn.logic;

import org.platanios.learn.logic.database.*;
import org.platanios.learn.logic.formula.Atom;
import org.platanios.learn.logic.formula.EntityType;
import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.formula.Variable;
import org.platanios.learn.logic.grounding.GroundPredicate;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DatabaseLogicManager<R> implements LogicManager<R> {
    private final Logic<R> logic;
    private final Class<R> valueClass;
    private final DatabaseManager databaseManager;

//    private final Map<Long, Set<Long>> entityTypeAllowedValues = new HashMap<>();

    public DatabaseLogicManager(Logic<R> logic, Class<R> valueClass) {
        this.logic = logic;
        this.valueClass = valueClass;
        this.databaseManager = new DatabaseManager.Builder().build();
    }

    public DatabaseLogicManager(Logic<R> logic, Class<R> valueClass, DatabaseManager databaseManager) {
        this.logic = logic;
        this.valueClass = valueClass;
        this.databaseManager = databaseManager;
    }

    public Logic<R> logic() {
        return logic;
    }

    public EntityType addEntityType(Set<Long> allowedValues) {
        return addEntityType(null, allowedValues);
    }

    public EntityType addEntityType(String name, Set<Long> allowedValues) {
        DatabaseEntityType databaseEntityType = databaseManager.addEntityType(name, allowedValues);
        return new EntityType(databaseEntityType.getId(),
                              databaseEntityType.getName(),
                              databaseEntityType.getAllowedValues()
                                      .stream()
                                      .map(DatabaseEntityTypeValue::getValue)
                                      .collect(Collectors.toSet()));
    }

    public Predicate addPredicate(List<EntityType> argumentTypes, boolean closed) {
        return addPredicate(null, argumentTypes, closed);
    }

    public Predicate addPredicate(String name, List<EntityType> argumentTypes, boolean closed) {
        DatabasePredicate databasePredicate = databaseManager.addPredicate(name, argumentTypes, closed);
        return new Predicate(databasePredicate.getId(),
                             databasePredicate.getName(),
                             databasePredicate.getArgumentTypes()
                                     .stream()
                                     .map(DatabasePredicateArgumentType::getArgumentType)
                                     .map(databaseEntityType ->
                                                  new EntityType(databaseEntityType.getId(),
                                                                 databaseEntityType.getName(),
                                                                 databaseEntityType.getAllowedValues()
                                                                         .stream()
                                                                         .map(DatabaseEntityTypeValue::getValue)
                                                                         .collect(Collectors.toSet())))
                                     .collect(Collectors.toList()));
    }

    public GroundPredicate<R> addGroundPredicate(Predicate predicate, List<Long> argumentAssignments) {
        return addGroundPredicate(predicate, argumentAssignments, null);
    }

    public GroundPredicate<R> addGroundPredicate(Predicate predicate, List<Long> argumentAssignments, R value) {
        DatabaseGroundPredicate databaseGroundPredicate = databaseManager.addGroundPredicate(
                predicate.getId(),
                argumentAssignments,
                value == null ? null : value.toString(),
                valueClass
        );
        return new GroundPredicate<>(databaseGroundPredicate.getId(), predicate, argumentAssignments, value);
    }

    public boolean checkIfGroundPredicateExists(Predicate predicate, List<Long> argumentAssignments) {
        return databaseManager.checkIfGroundPredicateExists(predicate.getId(), argumentAssignments);
    }

    public long getNumberOfGroundPredicates() {
        return databaseManager.getNumberOfGroundPredicates();
    }

    public List<GroundPredicate<R>> getGroundPredicates() {
        return databaseManager.getGroundPredicates();
    }

    public GroundPredicate<R> getGroundPredicate(long identifier) {
        return databaseManager.getGroundPredicate(identifier);
    }

    public GroundPredicate<R> getGroundPredicate(Predicate predicate, List<Long> variableAssignments) {
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
                .map(DatabaseEntityTypeValue::getValue)
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
    public R getPredicateAssignmentTruthValue(Predicate predicate, List<Long> variablesAssignment) {
        return databaseManager.getPredicateAssignmentTruthValue(predicate, variablesAssignment, logic);
    }

    public DatabaseManager.PartialGroundedFormula<R> getMatchingGroundPredicates(List<Atom> atoms) {
        return databaseManager.getMatchingGroundPredicates(atoms, logic);
    }
}
