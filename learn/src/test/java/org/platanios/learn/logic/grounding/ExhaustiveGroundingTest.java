package org.platanios.learn.logic.grounding;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.logic.InMemoryLogicManager;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.LukasiewiczLogic;
import org.platanios.learn.logic.formula.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ExhaustiveGroundingTest {
    @Test
    public void testAtomGrounding() {
        Set<Long> allowedValues = new HashSet<>();
        allowedValues.add((long) 0);
        allowedValues.add((long) 1);
        allowedValues.add((long) 2);
        allowedValues.add((long) 3);

        LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());
        EntityType personType = logicManager.addEntityType("{person}", allowedValues);
        Variable personA = new Variable(0, "A", personType);
        Variable personB = new Variable(1, "B", personType);
        List<EntityType> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Long> observedAssignments = new ArrayList<>();
        observedAssignments.add((long) 1);
        observedAssignments.add((long) 2);
        logicManager.addGroundPredicate(knowsPredicate, observedAssignments, 1.0);

        List<Variable> knowsVariables = new ArrayList<>(2);
        knowsVariables.add(personA);
        knowsVariables.add(personB);
        Formula knowsABAtom = new Atom(knowsPredicate, knowsVariables);

        ExhaustiveGrounding exhaustiveGrounding = new ExhaustiveGrounding(logicManager);
        exhaustiveGrounding.ground(knowsABAtom);

        Assert.assertEquals("knows(A, B)", knowsABAtom.toString());
    }

    @Test
    public void testNegationGrounding() {
        Set<Long> allowedValues = new HashSet<>();
        allowedValues.add((long) 0);
        allowedValues.add((long) 1);
        allowedValues.add((long) 2);
        allowedValues.add((long) 3);

        LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());
        EntityType personType = logicManager.addEntityType("{person}", allowedValues);
        Variable personA = new Variable(0, "A", personType);
        Variable personB = new Variable(1, "B", personType);
        List<EntityType> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Long> observedAssignments = new ArrayList<>();
        observedAssignments.add((long) 1);
        observedAssignments.add((long) 2);
        logicManager.addGroundPredicate(knowsPredicate, observedAssignments, 0.0);

        List<Variable> knowsVariables = new ArrayList<>(2);
        knowsVariables.add(personA);
        knowsVariables.add(personB);
        Formula knowsABAtom = new Negation(new Atom(knowsPredicate, knowsVariables));

        ExhaustiveGrounding exhaustiveGrounding = new ExhaustiveGrounding(logicManager);
        exhaustiveGrounding.ground(knowsABAtom);

        Assert.assertEquals("!knows(A, B)", knowsABAtom.toString());
    }

    @Test
    public void testDisjunctionGrounding() {
        Set<Long> allowedValues = new HashSet<>();
        allowedValues.add((long) 0);
        allowedValues.add((long) 1);
        allowedValues.add((long) 2);
        allowedValues.add((long) 3);

        LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());
        EntityType personType = logicManager.addEntityType("{person}", allowedValues);
        Variable personA = new Variable(0, "A", personType);
        Variable personB = new Variable(1, "B", personType);
        Variable personC = new Variable(2, "C", personType);
        List<EntityType> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Long> observed12Assignments = new ArrayList<>();
        observed12Assignments.add((long) 1);
        observed12Assignments.add((long) 2);
        List<Long> observed23Assignments = new ArrayList<>();
        observed23Assignments.add((long) 2);
        observed23Assignments.add((long) 3);
        logicManager.addGroundPredicate(knowsPredicate, observed12Assignments, 1.0);
        logicManager.addGroundPredicate(knowsPredicate, observed23Assignments, 1.0);

        List<Variable> knowsABVariables = new ArrayList<>(2);
        knowsABVariables.add(personA);
        knowsABVariables.add(personB);
        List<Variable> knowsBCVariables = new ArrayList<>(2);
        knowsBCVariables.add(personB);
        knowsBCVariables.add(personC);
        List<Formula> disjunctionComponents = new ArrayList<>(2);
        disjunctionComponents.add(new Atom(knowsPredicate, knowsABVariables));
        disjunctionComponents.add(new Atom(knowsPredicate, knowsBCVariables));
        Formula disjunctionFormula = new Disjunction(disjunctionComponents);

        ExhaustiveGrounding exhaustiveGrounding = new ExhaustiveGrounding(logicManager);
        exhaustiveGrounding.ground(disjunctionFormula);

        Assert.assertEquals("knows(A, B) | knows(B, C)", disjunctionFormula.toString());
    }

    @Test
    public void testDisjunctionWithNegationGrounding() {
        Set<Long> allowedValues = new HashSet<>();
        allowedValues.add((long) 0);
        allowedValues.add((long) 1);
        allowedValues.add((long) 2);
        allowedValues.add((long) 3);

        LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());
        EntityType personType = logicManager.addEntityType("{person}", allowedValues);
        Variable personA = new Variable(0, "A", personType);
        Variable personB = new Variable(1, "B", personType);
        Variable personC = new Variable(2, "C", personType);
        List<EntityType> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Long> observed12Assignments = new ArrayList<>();
        observed12Assignments.add((long) 1);
        observed12Assignments.add((long) 2);
        List<Long> observed23Assignments = new ArrayList<>();
        observed23Assignments.add((long) 2);
        observed23Assignments.add((long) 3);
        logicManager.addGroundPredicate(knowsPredicate, observed12Assignments, 1.0);
        logicManager.addGroundPredicate(knowsPredicate, observed23Assignments, 1.0);

        List<Variable> knowsABVariables = new ArrayList<>(2);
        knowsABVariables.add(personA);
        knowsABVariables.add(personB);
        List<Variable> knowsBCVariables = new ArrayList<>(2);
        knowsBCVariables.add(personB);
        knowsBCVariables.add(personC);
        List<Formula> disjunctionComponents = new ArrayList<>(2);
        disjunctionComponents.add(new Atom(knowsPredicate, knowsABVariables));
        disjunctionComponents.add(new Negation(new Atom(knowsPredicate, knowsBCVariables)));
        Formula disjunctionFormula = new Disjunction(disjunctionComponents);

        ExhaustiveGrounding exhaustiveGrounding = new ExhaustiveGrounding(logicManager);
        exhaustiveGrounding.ground(disjunctionFormula);

        Assert.assertEquals("knows(A, B) | !knows(B, C)", disjunctionFormula.toString());
    }
}
