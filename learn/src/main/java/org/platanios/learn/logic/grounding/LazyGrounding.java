package org.platanios.learn.logic.grounding;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.Atom;
import org.platanios.learn.logic.formula.Formula;
import org.platanios.learn.logic.formula.Negation;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LazyGrounding<R> extends Grounding<R> {
    private Set<ActivatedGroundedPredicate> activatedGroundedPredicates = new HashSet<>();

    public LazyGrounding(LogicManager<R> logicManager) {
        super(logicManager);
    }

    public LazyGrounding(LogicManager<R> logicManager,
                         Set<ActivatedGroundedPredicate> activatedGroundedPredicates) {
        super(logicManager);
        this.activatedGroundedPredicates = activatedGroundedPredicates;
    }

    public Set<ActivatedGroundedPredicate> getActivatedGroundedPredicates() {
        return activatedGroundedPredicates;
    }

    @Override
    public void ground(List<Formula> formulas) {
        while (true) {
            int previousNumberOfActivatedGroundedPredicates = activatedGroundedPredicates.size();
            for (int currentFormulaIndex = 0; currentFormulaIndex < formulas.size(); currentFormulaIndex++) {
                if (!groundedFormulas.containsKey(currentFormulaIndex))
                    groundedFormulas.put(currentFormulaIndex, new HashSet<>());
                ground(formulas.get(currentFormulaIndex));
                System.out.println("Generated " + groundedFormula.size() + " groundings for rule " + currentFormulaIndex); // TODO: Use a logger for this part.
                groundedFormulas.get(currentFormulaIndex).addAll(groundedFormula);
            }
            if (activatedGroundedPredicates.size() == previousNumberOfActivatedGroundedPredicates)
                break;
        }
    }

    @Override
    boolean pruneGroundingAndSetCurrentPredicateTruthValue(Formula formula,
                                                           Map<Long, Long> variableAssignments,
                                                           List<R> disjunctionComponentsSoFar) {
        if (currentPredicateTruthValue == null) // This is the important thing that PSL is doing while considering only the body variables.
            if (formula instanceof Negation
                    && !activatedGroundedPredicates.contains(
                    new ActivatedGroundedPredicate<>(
                            ((Atom) ((Negation) formula).getFormula()).getPredicate().getId(),
                            ((Atom) ((Negation) formula).getFormula()).getPredicateArguments().stream().map(variableAssignments::get).collect(Collectors.toList())
                    ))) {
                return true;
            } else
                disjunctionComponentsSoFar.add(logicManager.logic().falseValue());
        else if (formula instanceof Atom)
            disjunctionComponentsSoFar.add(currentPredicateTruthValue);
        else
            disjunctionComponentsSoFar.add(logicManager.logic().negation(currentPredicateTruthValue));
        currentPredicateTruthValue = logicManager.logic().disjunction(disjunctionComponentsSoFar);
        return logicManager.logic().isSatisfied(currentPredicateTruthValue);
    }

    @Override
    void onGroundedPredicateAddition(List<GroundPredicate<R>> groundedRule) {
        activatedGroundedPredicates.addAll(
                groundedRule.stream()
                        .filter(rule -> rule.getValue() == null)
                        .map(groundedPredicate ->
                                     new ActivatedGroundedPredicate<>(groundedPredicate.getPredicate().getId(),
                                                                      groundedPredicate.getPredicateArgumentsAssignment()))
                        .collect(Collectors.toList())
        );
    }

    public class ActivatedGroundedPredicate<V> {
        private final long predicateIdentifier;
        private final List<V> variableGroundings;

        private ActivatedGroundedPredicate(long predicateIdentifier, List<V> variableGroundings) {
            this.predicateIdentifier = predicateIdentifier;
            this.variableGroundings = variableGroundings;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        @SuppressWarnings("unchecked")
        public boolean equals(Object object) {
            if (!(object instanceof ActivatedGroundedPredicate))
                return false;
            if (object == this)
                return true;

            ActivatedGroundedPredicate<V> that = (ActivatedGroundedPredicate<V>) object;

            if (predicateIdentifier != that.predicateIdentifier)
                return false;
            if (!variableGroundings.equals(that.variableGroundings))
                return false;

            return true;
        }

        @Override
        public int hashCode() {
            HashCodeBuilder hashCodeBuilder = new HashCodeBuilder().append(predicateIdentifier);
            variableGroundings.forEach(hashCodeBuilder::append);
            return hashCodeBuilder.toHashCode();
        }
    }
}
