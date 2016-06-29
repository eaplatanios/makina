package makina.learn.logic.grounding;

import makina.learn.logic.formula.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.learn.logic.LogicManager;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class InMemoryLazyGrounding {
    private static Logger logger = LogManager.getFormatterLogger("In-Memory Lazy Grounding");

    final LogicManager logicManager;

    Map<Predicate, Set<GroundPredicate>> activatedGroundedPredicates = new HashMap<>(); // Predicate ID to set of grounded predicates with that predicate ID.
    Map<Long, List<Long>> groundedVariables = new HashMap<>(); // Maps from variable ID to list of grounded values -- the list is ordered in the same way as the groundedFormula list.
    List<List<GroundPredicate>> groundedFormula = new ArrayList<>(); // List of groundings -- each grounding is a set of grounded predicates representing the formula terms.
    List<Double> groundedFormulaTruthValues = new ArrayList<>();
    List<Boolean> formulaUnobservedVariableIndicators = new ArrayList<>();

    Map<Integer, Set<List<GroundPredicate>>> groundedFormulas = new HashMap<>();

    public InMemoryLazyGrounding(LogicManager logicManager) {
        this.logicManager = logicManager;
        for (GroundPredicate groundPredicate : logicManager.getGroundPredicates()) {
            if (!activatedGroundedPredicates.containsKey(groundPredicate.getPredicate()))
                activatedGroundedPredicates.put(groundPredicate.getPredicate(), new HashSet<>());
            activatedGroundedPredicates.get(groundPredicate.getPredicate()).add(groundPredicate);
        }
    }

    public List<Formula> ground(List<Formula> formulas) {
        return ground(formulas, false);
    }

    public List<Formula> ground(List<Formula> formulas, boolean logProgress) {
        List<Formula> preprocessedFormulas = new ArrayList<>();
        for (Formula formula : formulas) {
            formula = formula.toDisjunctiveNormalForm();
            List<Formula> disjunctionComponents = new ArrayList<>();
            if (formula instanceof Atom || formula instanceof Negation) {
                disjunctionComponents.add(formula);
            } else if (formula instanceof Disjunction) {
                disjunctionComponents.addAll(((Disjunction) formula).getComponents().stream().filter(innerFormula -> innerFormula instanceof Negation).collect(Collectors.toList()));
                disjunctionComponents.addAll(((Disjunction) formula).getComponents().stream().filter(innerFormula -> !(innerFormula instanceof Negation)).collect(Collectors.toList()));
            } else {
                throw new IllegalStateException("The formula being grounded was not converted to valid disjunctive " +
                                                        "normal form for some unknown reason.");
            }
            preprocessedFormulas.add(new Disjunction(disjunctionComponents));
        }
        int previousNumberOfActivatedGroundedPredicates = 0;
        for (Map.Entry<Predicate, Set<GroundPredicate>> activatedGroundedPredicate : activatedGroundedPredicates.entrySet())
            previousNumberOfActivatedGroundedPredicates += activatedGroundedPredicate.getValue().size();
        while (true) {
            for (int currentFormulaIndex = 0; currentFormulaIndex < preprocessedFormulas.size(); currentFormulaIndex++) {
                if (!groundedFormulas.containsKey(currentFormulaIndex))
                    groundedFormulas.put(currentFormulaIndex, new HashSet<>());
                ground(preprocessedFormulas.get(currentFormulaIndex));
                if (logProgress)
                    logger.info("Generated " + groundedFormula.size() + " groundings for rule " + currentFormulaIndex);
                groundedFormulas.get(currentFormulaIndex).addAll(groundedFormula);
            }
            int currentNumberOfActivatedGroundedPredicates = 0;
            for (Map.Entry<Predicate, Set<GroundPredicate>> activatedGroundedPredicate : activatedGroundedPredicates.entrySet())
                currentNumberOfActivatedGroundedPredicates += activatedGroundedPredicate.getValue().size();
            if (currentNumberOfActivatedGroundedPredicates == previousNumberOfActivatedGroundedPredicates)
                break;
            previousNumberOfActivatedGroundedPredicates = currentNumberOfActivatedGroundedPredicates;
        }
        return preprocessedFormulas;
    }

    private void ground(Formula formula) {
        groundedVariables = new HashMap<>();
        groundedFormula = new ArrayList<>();
        groundedFormulaTruthValues = new ArrayList<>();
        formulaUnobservedVariableIndicators = new ArrayList<>();
        ground(formula, 0);
//        List<List<T>> filteredPartialVariableGroundings = new ArrayList<>();
//        List<List<GroundedPredicate<T, R>>> filteredGroundedPredicates = new ArrayList<>();
//        for (int groundedRuleIndex = 0; groundedRuleIndex < groundedFormula.size(); groundedRuleIndex++) {
//            if (formulaUnobservedVariableIndicators.get(groundedRuleIndex)) {
//                filteredPartialVariableGroundings.add(groundedVariablesValues.get(groundedRuleIndex));
//                filteredGroundedPredicates.add(groundedFormula.get(groundedRuleIndex));
//            }
//        }
//        groundedVariablesValues = filteredPartialVariableGroundings;
//        groundedFormula = filteredGroundedPredicates;
    }

    @SuppressWarnings("unchecked")
    void ground(Formula formula, int callNumber) {
        if (callNumber != 0 && groundedFormula.size() == 0)
            return;
        if (formula instanceof Atom) {
            groundAtom((Atom) formula);
        } else if (formula instanceof Negation) {
            groundNegation((Negation) formula);
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

    private void groundNegation(Negation formula) {
        Predicate predicate = ((Atom) formula.getFormula()).getPredicate();
        List<Variable> variables = formula.getFormula().getOrderedVariables();
        if (groundedVariables.size() == 0) {
            for (Variable variable : variables)
                groundedVariables.put(variable.getId(), new ArrayList<>());
            for (GroundPredicate groundPredicate : activatedGroundedPredicates.getOrDefault(predicate, new HashSet<>())) {
                Double groundedPredicateValue = groundPredicate.getValue();
                Double currentTruthValue = groundedPredicateValue == null ? logicManager.logic().falseValue() : logicManager.logic().negation(groundedPredicateValue);
                if (logicManager.logic().isSatisfied(currentTruthValue))
                    continue;
                List<Long> variablesAssignment = groundPredicate.getArguments();
                for (int variableIndex = 0; variableIndex < variablesAssignment.size(); variableIndex++)
                    groundedVariables.get(variables.get(variableIndex).getId())
                            .add(variablesAssignment.get(variableIndex));
                List<GroundPredicate> temporaryList = new ArrayList<>();
                temporaryList.add(groundPredicate);
                groundedFormula.add(temporaryList);
                groundedFormulaTruthValues.add(currentTruthValue);
                formulaUnobservedVariableIndicators.add(false);
            }
        } else {
            Map<Long, List<Long>> newGroundedVariables = new HashMap<>();
            List<List<GroundPredicate>> newGroundedFormula = new ArrayList<>();
            List<Double> newGroundedFormulaTruthValues = new ArrayList<>();
            List<Boolean> newFormulaUnobservedVariableIndicators = new ArrayList<>();
            for (Variable variable : variables)
                newGroundedVariables.put(variable.getId(), new ArrayList<>());
            for (Long variableIdentifier : groundedVariables.keySet())
                newGroundedVariables.put(variableIdentifier, new ArrayList<>());
            for (int groundingIndex = 0; groundingIndex < groundedFormula.size(); groundingIndex++) {
                List<Long> variablesAssignmentTemplate = new ArrayList<>();
                for (Variable variable : variables) {
                    if (groundedVariables.containsKey(variable.getId()))
                        variablesAssignmentTemplate.add(groundedVariables.get(variable.getId()).get(groundingIndex));
                    else
                        variablesAssignmentTemplate.add(null);
                }
                for (GroundPredicate groundPredicate : activatedGroundedPredicates.getOrDefault(predicate, new HashSet<>())) {
                    boolean pruneGroundedPredicate = false;
                    List<Long> variablesAssignment = groundPredicate.getArguments();
                    for (int variableIndex = 0; variableIndex < variablesAssignment.size(); variableIndex++) {
                        if (!variablesAssignment.get(variableIndex).equals(variablesAssignmentTemplate.get(variableIndex))
                                && variablesAssignmentTemplate.get(variableIndex) != null) {
                            pruneGroundedPredicate = true;
                            break;
                        }
                    }
                    if (pruneGroundedPredicate)
                        continue;
                    List<Double> disjunctionComponentsSoFar = new ArrayList<>();
                    disjunctionComponentsSoFar.add(groundedFormulaTruthValues.get(groundingIndex));
                    Double currentPredicateTruthValue = groundPredicate.getValue();
                    if (currentPredicateTruthValue == null)
                        disjunctionComponentsSoFar.add(logicManager.logic().falseValue());
                    else
                        disjunctionComponentsSoFar.add(logicManager.logic().negation(currentPredicateTruthValue));
                    currentPredicateTruthValue = logicManager.logic().disjunction(disjunctionComponentsSoFar);
                    if (logicManager.logic().isSatisfied(currentPredicateTruthValue))
                        continue;
                    for (int variableIndex = 0; variableIndex < variablesAssignment.size(); variableIndex++)
                        if (variablesAssignmentTemplate.get(variableIndex) == null)
                            newGroundedVariables.get(variables.get(variableIndex).getId()).add(variablesAssignment.get(variableIndex));
                    for (Long variableIdentifier : newGroundedVariables.keySet())
                        if (groundedVariables.containsKey(variableIdentifier))
                            newGroundedVariables.get(variableIdentifier).add(groundedVariables.get(variableIdentifier).get(groundingIndex));
                    List<GroundPredicate> temporaryList = new ArrayList<>(groundedFormula.get(groundingIndex));
                    temporaryList.add(groundPredicate);
                    newGroundedFormula.add(temporaryList);
                    newGroundedFormulaTruthValues.add(currentPredicateTruthValue);
                    newFormulaUnobservedVariableIndicators.add(false);
                }
            }
            groundedVariables = newGroundedVariables;
            groundedFormula = newGroundedFormula;
            groundedFormulaTruthValues = newGroundedFormulaTruthValues;
            formulaUnobservedVariableIndicators = newFormulaUnobservedVariableIndicators;
        }
    }

    private void groundAtom(Atom formula) {
        formula.getVariables()
                .stream()
                .filter(argumentVariable -> !groundedVariables.containsKey(argumentVariable.getId()))
                .forEach(argumentVariable -> {
                    Map<Long, List<Long>> newGroundedVariables = new HashMap<>();
                    List<List<GroundPredicate>> newGroundedFormula = new ArrayList<>();
                    List<Double> newGroundedFormulaTruthValues = new ArrayList<>();
                    List<Boolean> newFormulaUnobservedVariableIndicators = new ArrayList<>();
                    newGroundedVariables.put(argumentVariable.getId(), new ArrayList<>());
                    if (groundedFormula.size() > 0) {
                        for (int index = 0; index < groundedFormula.size(); index++) {
                            for (long variableValue : logicManager.getVariableValues(argumentVariable)) {
                                for (Long variableIdentifier : groundedVariables.keySet())
                                    if (variableIdentifier != argumentVariable.getId()) {
                                        if (!newGroundedVariables.containsKey(variableIdentifier))
                                            newGroundedVariables.put(variableIdentifier, new ArrayList<>());
                                        newGroundedVariables.get(variableIdentifier).add(groundedVariables.get(variableIdentifier).get(index));
                                    }
                                newGroundedVariables.get(argumentVariable.getId()).add(variableValue);
                                newGroundedFormula.add(groundedFormula.get(index));
                                newGroundedFormulaTruthValues.add(groundedFormulaTruthValues.get(index));
                                newFormulaUnobservedVariableIndicators.add(formulaUnobservedVariableIndicators.get(index));
                            }
                        }
                    } else {
                        for (long variableValue : logicManager.getVariableValues(argumentVariable)) {
                            newGroundedVariables.get(argumentVariable.getId()).add(variableValue);
                            newGroundedFormula.add(new ArrayList<>());
                            newGroundedFormulaTruthValues.add(logicManager.logic().falseValue());
                            newFormulaUnobservedVariableIndicators.add(false);
                        }
                    }
                    groundedVariables = newGroundedVariables;
                    groundedFormula = newGroundedFormula;
                    groundedFormulaTruthValues = newGroundedFormulaTruthValues;
                    formulaUnobservedVariableIndicators = newFormulaUnobservedVariableIndicators;
                });
        Map<Long, List<Long>> newGroundedVariables = new HashMap<>();
        List<List<GroundPredicate>> newGroundedFormula = new ArrayList<>();
        List<Double> newGroundedFormulaTruthValues = new ArrayList<>();
        List<Boolean> newFormulaUnobservedVariableIndicators = new ArrayList<>();
        for (Long variableIdentifier : groundedVariables.keySet())
            newGroundedVariables.put(variableIdentifier, new ArrayList<>());
        for (int groundingIndex = 0; groundingIndex < groundedFormula.size(); groundingIndex++) {
            List<Long> newVariableGrounding = new ArrayList<>();
            Map<Long, Long> variableAssignments = new HashMap<>();
            for (Long variableIdentifier : groundedVariables.keySet()) {
                newVariableGrounding.add(groundedVariables.get(variableIdentifier).get(groundingIndex));
                variableAssignments.put(variableIdentifier, newVariableGrounding.get(newVariableGrounding.size() - 1));
            }
            List<Double> disjunctionComponentsSoFar = new ArrayList<>();
            disjunctionComponentsSoFar.add(groundedFormulaTruthValues.get(groundingIndex));
            Double currentPredicateTruthValue = formula.evaluate(logicManager, variableAssignments);
            if (currentPredicateTruthValue == null)
                disjunctionComponentsSoFar.add(logicManager.logic().falseValue());
            else
                disjunctionComponentsSoFar.add(currentPredicateTruthValue);
            currentPredicateTruthValue = logicManager.logic().disjunction(disjunctionComponentsSoFar);
            if (logicManager.logic().isSatisfied(currentPredicateTruthValue))
                continue;
            for (Long variableIdentifier : groundedVariables.keySet())
                newGroundedVariables.get(variableIdentifier).add(groundedVariables.get(variableIdentifier).get(groundingIndex));
            GroundPredicate groundPredicate;
            Predicate predicate = formula.getPredicate();
            newVariableGrounding = new ArrayList<>();
            for (Variable variable : formula.getOrderedVariables())
                newVariableGrounding.add(variableAssignments.get(variable.getId()));
            boolean unobservedVariable = formulaUnobservedVariableIndicators.get(groundingIndex);
            if (logicManager.checkIfGroundPredicateExists(predicate,
                                                          newVariableGrounding)) {
                groundPredicate = logicManager.getGroundPredicate(
                        predicate,
                        newVariableGrounding
                );
            } else {
                groundPredicate = logicManager.addGroundPredicate(
                        predicate,
                        newVariableGrounding
                );
            }
            List<GroundPredicate> temporaryList = new ArrayList<>(groundedFormula.get(groundingIndex));
            temporaryList.add(groundPredicate);
            newGroundedFormula.add(temporaryList);
            newGroundedFormulaTruthValues.add(currentPredicateTruthValue);
            newFormulaUnobservedVariableIndicators.add(unobservedVariable | groundPredicate.getValue() == null);
            if (!activatedGroundedPredicates.containsKey(groundPredicate.getPredicate()))
                activatedGroundedPredicates.put(groundPredicate.getPredicate(), new HashSet<>());
            activatedGroundedPredicates.get(groundPredicate.getPredicate()).add(groundPredicate);
        }
        groundedVariables = newGroundedVariables;
        groundedFormula = newGroundedFormula;
        groundedFormulaTruthValues = newGroundedFormulaTruthValues;
        formulaUnobservedVariableIndicators = newFormulaUnobservedVariableIndicators;
    }

    public Map<Integer, Set<List<GroundPredicate>>> getGroundedFormulas() {
        return groundedFormulas;
    }
}
