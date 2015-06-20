package org.platanios.learn.logic;

import org.platanios.learn.logic.formula.EntityType;
import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.formula.Variable;
import org.platanios.learn.logic.grounding.GroundPredicate;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface LogicManager<R> {
    Logic<R> logic();
    EntityType addEntityType(List<Long> allowedValues);
    EntityType addEntityType(String name, List<Long> allowedValues);
    Predicate addPredicate(List<EntityType> argumentTypes, boolean closed);
    Predicate addPredicate(String name, List<EntityType> argumentTypes, boolean closed);
    GroundPredicate<R> addGroundPredicate(Predicate predicate, List<Long> argumentAssignments);
    GroundPredicate<R> addGroundPredicate(Predicate predicate, List<Long> argumentAssignments, R value);
    boolean checkIfGroundPredicateExists(Predicate predicate, List<Long> argumentAssignments);
    long getNumberOfGroundPredicates();
    List<GroundPredicate<R>> getGroundPredicates();
    GroundPredicate<R> getGroundPredicate(long identifier);
    GroundPredicate<R> getGroundPredicate(Predicate predicate, List<Long> argumentAssignments);
    long getNumberOfEntityTypes();
    EntityType getEntityType(long identifier);
    EntityType getEntityType(String name);
    List<Long> getVariableValues(Variable variable);
    Predicate getPredicate(long identifier);
    Predicate getPredicate(String name);
    List<Predicate> getClosedPredicates();
    R getPredicateAssignmentTruthValue(Predicate predicate, List<Long> argumentAssignments);
}
