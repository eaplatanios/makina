package org.platanios.learn.logic.database;

import org.junit.Test;
import org.platanios.learn.logic.DatabaseLogicManager;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.LukasiewiczLogic;
import org.platanios.learn.logic.formula.Atom;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DatabaseManagerTest {
    @Test
    public void testGetMatchingGroundPredicates() {
        Set<Long> allowedValues = new HashSet<>();
        allowedValues.add((long) 0);
        allowedValues.add((long) 1);
        allowedValues.add((long) 2);
        allowedValues.add((long) 3);

        DatabaseManager databaseManager = new DatabaseManager.Builder().build();
        LogicManager<Double> logicManager = new DatabaseLogicManager<>(new LukasiewiczLogic(),
                                                                       Double.class,
                                                                       databaseManager);
        org.platanios.learn.logic.formula.EntityType personType = logicManager.addEntityType("{person}", allowedValues);
        org.platanios.learn.logic.formula.Variable personA = new org.platanios.learn.logic.formula.Variable((long) 0, "A", personType);
        org.platanios.learn.logic.formula.Variable personB = new org.platanios.learn.logic.formula.Variable((long) 1, "B", personType);
        org.platanios.learn.logic.formula.Variable personC = new org.platanios.learn.logic.formula.Variable((long) 2, "C", personType);
        List<org.platanios.learn.logic.formula.EntityType> predicateArgumentTypes = new ArrayList<>(2);
        predicateArgumentTypes.add(personType);
        predicateArgumentTypes.add(personType);
        org.platanios.learn.logic.formula.Predicate knowsPredicate = logicManager.addPredicate("knows", predicateArgumentTypes, false);
        List<Long> knowsObservedAssignments = new ArrayList<>(2);
        knowsObservedAssignments.add((long) 1);
        knowsObservedAssignments.add((long) 2);
        logicManager.addGroundPredicate(knowsPredicate, knowsObservedAssignments, 1.0);
        org.platanios.learn.logic.formula.Predicate trustsPredicate = logicManager.addPredicate("trusts", predicateArgumentTypes, false);
        List<Long> trustsObservedAssignments = new ArrayList<>(2);
        trustsObservedAssignments.add((long) 2);
        trustsObservedAssignments.add((long) 3);
        logicManager.addGroundPredicate(trustsPredicate, trustsObservedAssignments, 1.0);

        List<org.platanios.learn.logic.formula.Variable> knowsVariables = new ArrayList<>(2);
        knowsVariables.add(personA);
        knowsVariables.add(personB);
        Atom knowsABAtom = new Atom(knowsPredicate, knowsVariables);
        List<org.platanios.learn.logic.formula.Variable> trustsVariables = new ArrayList<>(2);
        trustsVariables.add(personB);
        trustsVariables.add(personC);
        Atom trustsBCAtom = new Atom(trustsPredicate, trustsVariables);
        List<Atom> atomsList = new ArrayList<>(2);
        atomsList.add(knowsABAtom);
        atomsList.add(trustsBCAtom);

        databaseManager.getMatchingGroundPredicates(atomsList, logicManager.logic());
    }
}
