package org.platanios.learn.logic.grounding;

import org.junit.Assert;
import org.junit.Test;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.LukasiewiczLogic;
import org.platanios.learn.logic.formula.*;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ExhaustiveGroundingTest {
    @Test
    public void testAtomGrounding() {
        List<Integer> allowedValues = new ArrayList<>();
        allowedValues.add(0);
        allowedValues.add(1);
        allowedValues.add(2);
        allowedValues.add(3);

        LogicManager<Integer, Double> logicManager = new LogicManager<>(new LukasiewiczLogic());
        VariableType<Integer> personType = logicManager.addVariableType("{person}", Integer.class);
        Variable<Integer> personA = logicManager.addVariable("A", allowedValues, personType);
        Variable<Integer> personB = logicManager.addVariable("B", allowedValues, personType);
        List<VariableType<Integer>> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate<Integer> knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Integer> observedAssignments = new ArrayList<>();
        observedAssignments.add(1);
        observedAssignments.add(2);
        logicManager.addGroundedPredicate(knowsPredicate, observedAssignments, 1.0);

        List<Variable<Integer>> knowsVariables = new ArrayList<>(2);
        knowsVariables.add(personA);
        knowsVariables.add(personB);
        Formula<Integer> knowsABAtom = new Atom<>(knowsPredicate, knowsVariables);

        ExhaustiveGrounding<Integer, Double> exhaustiveGrounding = new ExhaustiveGrounding<>(logicManager);
        exhaustiveGrounding.ground(knowsABAtom);

        Assert.assertEquals("knows(A, B)", knowsABAtom.toString());
    }

    @Test
    public void testNegationGrounding() {
        List<Integer> allowedValues = new ArrayList<>();
        allowedValues.add(0);
        allowedValues.add(1);
        allowedValues.add(2);
        allowedValues.add(3);

        LogicManager<Integer, Double> logicManager = new LogicManager<>(new LukasiewiczLogic());
        VariableType<Integer> personType = logicManager.addVariableType("{person}", Integer.class);
        Variable<Integer> personA = logicManager.addVariable("A", allowedValues, personType);
        Variable<Integer> personB = logicManager.addVariable("B", allowedValues, personType);
        List<VariableType<Integer>> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate<Integer> knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Integer> observedAssignments = new ArrayList<>();
        observedAssignments.add(1);
        observedAssignments.add(2);
        logicManager.addGroundedPredicate(knowsPredicate, observedAssignments, 0.0);

        List<Variable<Integer>> knowsVariables = new ArrayList<>(2);
        knowsVariables.add(personA);
        knowsVariables.add(personB);
        Formula<Integer> knowsABAtom = new Negation<>(new Atom<>(knowsPredicate, knowsVariables));

        ExhaustiveGrounding<Integer, Double> exhaustiveGrounding = new ExhaustiveGrounding<>(logicManager);
        exhaustiveGrounding.ground(knowsABAtom);

        Assert.assertEquals("knows(A, B)", knowsABAtom.toString());
    }

    @Test
    public void testDisjunctionGrounding() {
        List<Integer> allowedValues = new ArrayList<>();
        allowedValues.add(0);
        allowedValues.add(1);
        allowedValues.add(2);
        allowedValues.add(3);

        LogicManager<Integer, Double> logicManager = new LogicManager<>(new LukasiewiczLogic());
        VariableType<Integer> personType = logicManager.addVariableType("{person}", Integer.class);
        Variable<Integer> personA = logicManager.addVariable("A", allowedValues, personType);
        Variable<Integer> personB = logicManager.addVariable("B", allowedValues, personType);
        Variable<Integer> personC = logicManager.addVariable("C", allowedValues, personType);
        List<VariableType<Integer>> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate<Integer> knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Integer> observed12Assignments = new ArrayList<>();
        observed12Assignments.add(1);
        observed12Assignments.add(2);
        List<Integer> observed23Assignments = new ArrayList<>();
        observed23Assignments.add(2);
        observed23Assignments.add(3);
        logicManager.addGroundedPredicate(knowsPredicate, observed12Assignments, 1.0);
        logicManager.addGroundedPredicate(knowsPredicate, observed23Assignments, 1.0);

        List<Variable<Integer>> knowsABVariables = new ArrayList<>(2);
        knowsABVariables.add(personA);
        knowsABVariables.add(personB);
        List<Variable<Integer>> knowsBCVariables = new ArrayList<>(2);
        knowsBCVariables.add(personB);
        knowsBCVariables.add(personC);
        List<Formula<Integer>> disjunctionComponents = new ArrayList<>(2);
        disjunctionComponents.add(new Atom<>(knowsPredicate, knowsABVariables));
        disjunctionComponents.add(new Atom<>(knowsPredicate, knowsBCVariables));
        Formula<Integer> disjunctionFormula = new Disjunction<>(disjunctionComponents);

        ExhaustiveGrounding<Integer, Double> exhaustiveGrounding = new ExhaustiveGrounding<>(logicManager);
        exhaustiveGrounding.ground(disjunctionFormula);

        Assert.assertEquals("knows(A, B)", disjunctionFormula.toString());
    }

    @Test
    public void testDisjunctionWithNegationGrounding() {
        List<Integer> allowedValues = new ArrayList<>();
        allowedValues.add(0);
        allowedValues.add(1);
        allowedValues.add(2);
        allowedValues.add(3);

        LogicManager<Integer, Double> logicManager = new LogicManager<>(new LukasiewiczLogic());
        VariableType<Integer> personType = logicManager.addVariableType("{person}", Integer.class);
        Variable<Integer> personA = logicManager.addVariable("A", allowedValues, personType);
        Variable<Integer> personB = logicManager.addVariable("B", allowedValues, personType);
        Variable<Integer> personC = logicManager.addVariable("C", allowedValues, personType);
        List<VariableType<Integer>> knowsArgumentTypes = new ArrayList<>(2);
        knowsArgumentTypes.add(personType);
        knowsArgumentTypes.add(personType);
        Predicate<Integer> knowsPredicate = logicManager.addPredicate("knows", knowsArgumentTypes, false);
        List<Integer> observed12Assignments = new ArrayList<>();
        observed12Assignments.add(1);
        observed12Assignments.add(2);
        List<Integer> observed23Assignments = new ArrayList<>();
        observed23Assignments.add(2);
        observed23Assignments.add(3);
        logicManager.addGroundedPredicate(knowsPredicate, observed12Assignments, 1.0);
        logicManager.addGroundedPredicate(knowsPredicate, observed23Assignments, 1.0);

        List<Variable<Integer>> knowsABVariables = new ArrayList<>(2);
        knowsABVariables.add(personA);
        knowsABVariables.add(personB);
        List<Variable<Integer>> knowsBCVariables = new ArrayList<>(2);
        knowsBCVariables.add(personB);
        knowsBCVariables.add(personC);
        List<Formula<Integer>> disjunctionComponents = new ArrayList<>(2);
        disjunctionComponents.add(new Atom<>(knowsPredicate, knowsABVariables));
        disjunctionComponents.add(new Negation<>(new Atom<>(knowsPredicate, knowsBCVariables)));
        Formula<Integer> disjunctionFormula = new Disjunction<>(disjunctionComponents);

        ExhaustiveGrounding<Integer, Double> exhaustiveGrounding = new ExhaustiveGrounding<>(logicManager);
        exhaustiveGrounding.ground(disjunctionFormula);

        Assert.assertEquals("knows(A, B)", disjunctionFormula.toString());
    }
}
