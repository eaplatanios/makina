package makina.learn.logic;

import makina.learn.logic.formula.EntityType;
import makina.learn.logic.formula.Predicate;
import makina.learn.logic.formula.Variable;
import makina.learn.logic.grounding.GroundPredicate;

import java.util.List;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface LogicManager {
    Logic logic();
    EntityType addEntityType(Set<Long> allowedValues);
    EntityType addEntityType(String name, Set<Long> allowedValues);
    Predicate addPredicate(List<EntityType> argumentTypes, boolean closed);
    Predicate addPredicate(String name, List<EntityType> argumentTypes, boolean closed);
    GroundPredicate addGroundPredicate(Predicate predicate, List<Long> argumentAssignments);
    GroundPredicate addGroundPredicate(Predicate predicate, List<Long> argumentAssignments, Double value);
    GroundPredicate addOrReplaceGroundPredicate(Predicate predicate, List<Long> argumentAssignments);
    GroundPredicate addOrReplaceGroundPredicate(Predicate predicate, List<Long> argumentAssignments, Double value);
    GroundPredicate removeGroundPredicate(Predicate predicate, List<Long> argumentAssignments);
    boolean checkIfGroundPredicateExists(Predicate predicate, List<Long> argumentAssignments);
    long getNumberOfGroundPredicates();
    List<GroundPredicate> getGroundPredicates();
    GroundPredicate getGroundPredicate(long identifier);
    GroundPredicate getGroundPredicate(Predicate predicate, List<Long> argumentAssignments);
    long getNumberOfEntityTypes();
    EntityType getEntityType(String name);
    Set<Long> getVariableValues(Variable variable);
    Predicate getPredicate(String name);
    List<Predicate> getClosedPredicates();
    Double getPredicateAssignmentTruthValue(Predicate predicate, List<Long> argumentAssignments);
}
