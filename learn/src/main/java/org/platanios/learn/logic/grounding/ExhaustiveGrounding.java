package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ExhaustiveGrounding<T, R> {
    private final LogicManager<T, R> logicManager;

    private List<R> groundingTruthValues = new ArrayList<>();
    private List<Variable<T>> groundedVariables = new ArrayList<>();
    private List<List<T>> partialVariableGroundings = new ArrayList<>();
    private List<List<GroundedPredicate<T, R>>> groundedPredicates = new ArrayList<>();

    public ExhaustiveGrounding(LogicManager<T, R> logicManager) {
        this.logicManager = logicManager;
    }

    public void ground(Formula<T> formula) {
        ground(formula.toDisjunctiveNormalForm(), 0);
    }

    @SuppressWarnings("unchecked")
    private void ground(Formula<T> formula, int callNumber) {
        if (callNumber != 0 && partialVariableGroundings.size() == 0)
            return;
        if (formula instanceof Atom || formula instanceof Negation) {
            List<Variable<T>> argumentVariables = formula
                    .getVariables()
                    .stream()
                    .collect(Collectors.toList());
            List<Variable<T>> newGroundedVariables = new ArrayList<>(groundedVariables);
            List<R> candidateGroundingTruthValues = new ArrayList<>(groundingTruthValues);
            List<List<T>> candidateVariableGroundings = new ArrayList<>(partialVariableGroundings);
            List<List<GroundedPredicate<T, R>>> candidateGroundedPredicates = new ArrayList<>(groundedPredicates);
            argumentVariables.stream()
                    .filter(argumentVariable -> !groundedVariables.contains(argumentVariable))
                    .forEach(argumentVariable -> {
                        candidateGroundingTruthValues.clear();
                        candidateVariableGroundings.clear();
                        candidateGroundedPredicates.clear();
                        newGroundedVariables.add(argumentVariable);
                        if (partialVariableGroundings.size() > 0) {
                            for (int index = 0; index < partialVariableGroundings.size(); index++) {
                                for (T variableValue : logicManager.getVariableValues(argumentVariable)) {
                                    List<T> variableGrounding = new ArrayList<>(partialVariableGroundings.get(index));
                                    variableGrounding.add(variableValue);
                                    candidateGroundingTruthValues.add(groundingTruthValues.get(index));
                                    candidateVariableGroundings.add(variableGrounding);
                                    candidateGroundedPredicates.add(new ArrayList<>(groundedPredicates.get(index)));
                                }
                            }
                        } else {
                            for (T variableValue : logicManager.getVariableValues(argumentVariable)) {
                                List<T> variableGrounding = new ArrayList<>();
                                variableGrounding.add(variableValue);
                                candidateGroundingTruthValues.add(logicManager.logic().falseValue());
                                candidateVariableGroundings.add(variableGrounding);
                                candidateGroundedPredicates.add(new ArrayList<>());
                            }
                        }
                        groundingTruthValues = new ArrayList<>(candidateGroundingTruthValues);
                        partialVariableGroundings = new ArrayList<>(candidateVariableGroundings);
                        groundedPredicates = new ArrayList<>(candidateGroundedPredicates);
                    });
            List<R> truthValues = new ArrayList<>(groundingTruthValues);
            groundedVariables = newGroundedVariables;
            groundingTruthValues = new ArrayList<>();
            partialVariableGroundings = new ArrayList<>();
            groundedPredicates = new ArrayList<>();
            for (int candidateIndex = 0; candidateIndex < candidateGroundedPredicates.size(); candidateIndex++) {
                List<T> candidateVariableGrounding = candidateVariableGroundings.get(candidateIndex);
                Map<Variable<T>, T> variableAssignments = new HashMap<>();
                for (int variableIndex = 0; variableIndex < groundedVariables.size(); variableIndex++)
                    variableAssignments.put(groundedVariables.get(variableIndex), candidateVariableGrounding.get(variableIndex));
                List<R> disjunctionComponents = new ArrayList<>();
                disjunctionComponents.add(truthValues.get(candidateIndex));
                R truthValue;
                if (formula instanceof Atom)
                    truthValue = formula.evaluate(logicManager, variableAssignments);
                else
                    truthValue = ((Negation<T>) formula).getFormula().evaluate(logicManager, variableAssignments);
                if (truthValue == null)
                    disjunctionComponents.add(logicManager.logic().falseValue());
                else if (formula instanceof Atom)
                    disjunctionComponents.add(truthValue);
                else
                    disjunctionComponents.add(logicManager.logic().negation(truthValue));
                truthValue = logicManager.logic().disjunction(disjunctionComponents);
                if (!logicManager.logic().isSatisfied(truthValue)) {
                    groundingTruthValues.add(truthValue);
                    partialVariableGroundings.add(candidateVariableGrounding);
                    GroundedPredicate<T, R> groundedPredicate;
                    Predicate<T> predicate;
                    if (formula instanceof Atom)
                        predicate = ((Atom<T>) formula).getPredicate();
                    else
                        predicate = ((Atom<T>) ((Negation<T>) formula).getFormula()).getPredicate();
                    candidateVariableGrounding = new ArrayList<>();
                    for (Variable<T> variable : formula.getOrderedVariables())
                        candidateVariableGrounding.add(variableAssignments.get(variable));
                    if (logicManager.predicateGroundingExists(predicate,
                                                              candidateVariableGrounding)) {
                        groundedPredicate = logicManager.getGroundedPredicate(
                                predicate,
                                candidateVariableGrounding
                        );
                    } else {
                        groundedPredicate = logicManager.addGroundedPredicate(
                                predicate,
                                candidateVariableGrounding
                        );
                    }
                    candidateGroundedPredicates.get(candidateIndex).add(groundedPredicate);
                    groundedPredicates.add(candidateGroundedPredicates.get(candidateIndex));
                }
            }
        } else if (formula instanceof Conjunction) {
            if (callNumber == 0)
                throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
                                                        "normal form for some unknown reason.");

            throw new UnsupportedOperationException();
//            int numberOfComponents = ((Conjunction<T>) formula).getNumberOfComponents();
//            for (int componentIndex = 0; componentIndex < numberOfComponents; componentIndex++) {
//                Formula<T> componentFormula = ((Conjunction<T>) formula).getComponent(componentIndex);
//                if (componentFormula instanceof Atom || componentFormula instanceof Negation) {
//                    groundAtomOrNegation(formula, truthValues, candidateVariableGroundings, candidatePredicateGroundings);
//                } else {
//                    throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
//                                                            "normal form for some unknown reason.");
//                }
//            }
        } else if (formula instanceof Disjunction) {
            if (callNumber != 0)
                throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
                                                        "normal form for some unknown reason.");

            int numberOfComponents = ((Disjunction<T>) formula).getNumberOfComponents();
            for (int componentIndex = 0; componentIndex < numberOfComponents; componentIndex++) {
                Formula<T> componentFormula = ((Disjunction<T>) formula).getComponent(componentIndex);
                if (componentFormula instanceof Atom || componentFormula instanceof Negation) {
                    ground(componentFormula, callNumber++);
                } else {
                    throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
                                                            "normal form for some unknown reason.");
                }
            }
        } else {
            throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
                                                    "normal form for some unknown reason.");
        }
    }

    public List<List<GroundedPredicate<T, R>>> getGroundedPredicates() {
        return groundedPredicates;
    }
}
