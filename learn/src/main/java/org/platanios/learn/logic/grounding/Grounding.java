package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Grounding<R> {
    final LogicManager<R> logicManager;

    List<R> groundingTruthValues = new ArrayList<>();
    List<Variable> groundedVariables = new ArrayList<>();
    List<List<Long>> partialVariableGroundings = new ArrayList<>();
    List<List<GroundPredicate<R>>> groundedFormula = new ArrayList<>();
    Map<Integer, Set<List<GroundPredicate<R>>>> groundedFormulas = new HashMap<>();
    List<Boolean> ruleUnobservedVariableIndicators = new ArrayList<>();
    R currentPredicateTruthValue;

    public Grounding(LogicManager<R> logicManager) {
        this.logicManager = logicManager;
    }

    public abstract void ground(List<Formula> formulas);

    public void ground(Formula formula) {
        groundingTruthValues = new ArrayList<>();
        groundedVariables = new ArrayList<>();
        partialVariableGroundings = new ArrayList<>();
        groundedFormula = new ArrayList<>();
        ruleUnobservedVariableIndicators = new ArrayList<>();
        currentPredicateTruthValue = logicManager.logic().falseValue();
        ground(formula.toDisjunctiveNormalForm(), 0);
        List<List<Long>> filteredPartialVariableGroundings = new ArrayList<>();
        List<List<GroundPredicate<R>>> filteredGroundedPredicates = new ArrayList<>();
        for (int groundedRuleIndex = 0; groundedRuleIndex < groundedFormula.size(); groundedRuleIndex++) {
            if (ruleUnobservedVariableIndicators.get(groundedRuleIndex)) {
                filteredPartialVariableGroundings.add(partialVariableGroundings.get(groundedRuleIndex));
                filteredGroundedPredicates.add(groundedFormula.get(groundedRuleIndex));
            }
        }
        partialVariableGroundings = filteredPartialVariableGroundings;
        groundedFormula = filteredGroundedPredicates;
    }

    @SuppressWarnings("unchecked")
    void ground(Formula formula, int callNumber) {
        if (callNumber != 0 && partialVariableGroundings.size() == 0)
            return;
        if (formula instanceof Atom || formula instanceof Negation) {
            List<Variable> argumentVariables = formula
                    .getVariables()
                    .stream()
                    .collect(Collectors.toList());
            List<Variable> newGroundedVariables = new ArrayList<>(groundedVariables);
            List<R> candidateGroundingTruthValues = new ArrayList<>(groundingTruthValues);
            List<List<Long>> candidateVariableGroundings = new ArrayList<>(partialVariableGroundings);
            List<List<GroundPredicate<R>>> candidateGroundedFormula = new ArrayList<>(groundedFormula);
            List<Boolean> candidateRuleUnobservedVariableIndicators = new ArrayList<>(ruleUnobservedVariableIndicators);
            argumentVariables.stream()
                    .filter(argumentVariable -> !groundedVariables.contains(argumentVariable))
                    .forEach(argumentVariable -> {
                        candidateGroundingTruthValues.clear();
                        candidateVariableGroundings.clear();
                        candidateGroundedFormula.clear();
                        candidateRuleUnobservedVariableIndicators.clear();
                        newGroundedVariables.add(argumentVariable);
                        if (partialVariableGroundings.size() > 0) {
                            for (int index = 0; index < partialVariableGroundings.size(); index++) {
                                for (long variableValue : logicManager.getVariableValues(argumentVariable)) {
                                    List<Long> variableGrounding = new ArrayList<>(partialVariableGroundings.get(index));
                                    variableGrounding.add(variableValue);
                                    candidateGroundingTruthValues.add(groundingTruthValues.get(index));
                                    candidateVariableGroundings.add(variableGrounding);
                                    candidateGroundedFormula.add(new ArrayList<>(groundedFormula.get(index)));
                                    candidateRuleUnobservedVariableIndicators.add(ruleUnobservedVariableIndicators.get(index));
                                }
                            }
                        } else {
                            for (long variableValue : logicManager.getVariableValues(argumentVariable)) {
                                List<Long> variableGrounding = new ArrayList<>();
                                variableGrounding.add(variableValue);
                                candidateGroundingTruthValues.add(logicManager.logic().falseValue());
                                candidateVariableGroundings.add(variableGrounding);
                                candidateGroundedFormula.add(new ArrayList<>());
                                candidateRuleUnobservedVariableIndicators.add(false);
                            }
                        }
                        groundingTruthValues = new ArrayList<>(candidateGroundingTruthValues);
                        partialVariableGroundings = new ArrayList<>(candidateVariableGroundings);
                        groundedFormula = new ArrayList<>(candidateGroundedFormula);
                        ruleUnobservedVariableIndicators = new ArrayList<>(candidateRuleUnobservedVariableIndicators);
                    });
            List<R> truthValues = new ArrayList<>(groundingTruthValues);
            groundedVariables = newGroundedVariables;
            groundingTruthValues = new ArrayList<>();
            partialVariableGroundings = new ArrayList<>();
            groundedFormula = new ArrayList<>();
            ruleUnobservedVariableIndicators = new ArrayList<>();
            for (int candidateIndex = 0; candidateIndex < candidateGroundedFormula.size(); candidateIndex++) {
                List<Long> candidateVariableGrounding = candidateVariableGroundings.get(candidateIndex);
                Map<Long, Long> variableAssignments = new HashMap<>();
                for (int variableIndex = 0; variableIndex < groundedVariables.size(); variableIndex++)
                    variableAssignments.put(groundedVariables.get(variableIndex).getId(), candidateVariableGrounding.get(variableIndex));
                List<R> disjunctionComponentsSoFar = new ArrayList<>();
                disjunctionComponentsSoFar.add(truthValues.get(candidateIndex));
                if (formula instanceof Atom)
                    currentPredicateTruthValue = formula.evaluate(logicManager, variableAssignments);
                else
                    currentPredicateTruthValue = ((Negation) formula).getFormula().evaluate(logicManager, variableAssignments);
                if (!pruneGroundingAndSetCurrentPredicateTruthValue(formula, variableAssignments, disjunctionComponentsSoFar)) {
                    groundingTruthValues.add(currentPredicateTruthValue);
                    partialVariableGroundings.add(candidateVariableGrounding);
                    GroundPredicate<R> groundPredicate;
                    Predicate predicate;
                    if (formula instanceof Atom)
                        predicate = ((Atom) formula).getPredicate();
                    else
                        predicate = ((Atom) ((Negation) formula).getFormula()).getPredicate();
                    candidateVariableGrounding = new ArrayList<>();
                    for (Variable variable : formula.getOrderedVariables())
                        candidateVariableGrounding.add(variableAssignments.get(variable.getId()));
                    boolean unobservedVariable = candidateRuleUnobservedVariableIndicators.get(candidateIndex);
                    if (logicManager.checkIfGroundPredicateExists(predicate,
                                                                  candidateVariableGrounding)) {
                        groundPredicate = logicManager.getGroundPredicate(
                                predicate,
                                candidateVariableGrounding
                        );
                    } else {
                        groundPredicate = logicManager.addGroundPredicate(
                                predicate,
                                candidateVariableGrounding
                        );
                    }
                    candidateGroundedFormula.get(candidateIndex).add(groundPredicate);
                    groundedFormula.add(candidateGroundedFormula.get(candidateIndex));
                    ruleUnobservedVariableIndicators.add(unobservedVariable | groundPredicate.getValue() == null);
                    onGroundedPredicateAddition(candidateGroundedFormula.get(candidateIndex));
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

            int numberOfComponents = ((Disjunction) formula).getNumberOfComponents();
            for (int componentIndex = 0; componentIndex < numberOfComponents; componentIndex++) {
                Formula componentFormula = ((Disjunction) formula).getComponent(componentIndex);
                if (componentFormula instanceof Atom || componentFormula instanceof Negation)
                    ground(componentFormula, callNumber++);
                else
                    throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
                                                            "normal form for some unknown reason.");
            }
        } else {
            throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
                                                    "normal form for some unknown reason.");
        }
    }

    abstract boolean pruneGroundingAndSetCurrentPredicateTruthValue(Formula formula,
                                                                    Map<Long, Long> variableAssignments,
                                                                    List<R> disjunctionComponentsSoFar);

    void onGroundedPredicateAddition(List<GroundPredicate<R>> groundPredicate) {

    }

    public List<List<GroundPredicate<R>>> getGroundedFormula() {
        return groundedFormula;
    }

    public Map<Integer, Set<List<GroundPredicate<R>>>> getGroundedFormulas() {
        return groundedFormulas;
    }
}
