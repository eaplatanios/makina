package org.platanios.experiment.psl;

import com.google.common.collect.*;
import com.google.common.primitives.Booleans;
import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.Utilities;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.*;
import org.platanios.learn.logic.grounding.ExhaustiveGrounding;
import org.platanios.learn.logic.grounding.GroundedPredicate;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;
import org.platanios.learn.optimization.NewtonSolver;
import org.platanios.learn.optimization.constraint.AbstractConstraint;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.MaxFunction;
import org.platanios.learn.optimization.function.SumFunction;
import org.platanios.learn.optimization.linesearch.NoLineSearch;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class ProbabilisticSoftLogicProblem {
    private final BiMap<Integer, Integer> externalToInternalIndexesMapping;
    private final ProbabilisticSoftLogicFunction objectiveFunction;
    private final ImmutableSet<Constraint> constraints;
    private final Map<Integer, CholeskyDecomposition> subProblemCholeskyFactors = new HashMap<>();
    private final Map<Integer, List<Integer>> externalPredicateIdToTerms;

    @Override
    public boolean equals(Object other) {

        if (!(other instanceof ProbabilisticSoftLogicProblem)) {
            return false;
        }
        if (other == this) {
            return true;
        }

        ProbabilisticSoftLogicProblem rhs = (ProbabilisticSoftLogicProblem) other;

        return new EqualsBuilder()
                .append(this.externalToInternalIndexesMapping, rhs.externalToInternalIndexesMapping)
                .append(this.objectiveFunction, rhs.objectiveFunction)
                .append(this.constraints, rhs.constraints)
                .append(this.subProblemCholeskyFactors, rhs.subProblemCholeskyFactors)
                .append(this.externalPredicateIdToTerms, rhs.externalPredicateIdToTerms)
                .isEquals();

    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 31)
                .append(this.externalToInternalIndexesMapping)
                .append(this.objectiveFunction)
                .append(this.constraints)
                .append(this.subProblemCholeskyFactors)
                .append(this.externalPredicateIdToTerms)
                .toHashCode();
    }

    public static class Predicate implements Serializable {

        public Predicate(String name, ImmutableList<String> arguments, boolean isNegated) {
            this.Name = name;
            this.Arguments = arguments;
            this.IsNegated = isNegated;
        }

        public final String Name;
        public final ImmutableList<String> Arguments;
        public final boolean IsNegated;

        @Override
        public boolean equals(Object other) {
            if (other == this) {
                return true;
            }

            if (!(other instanceof Predicate)) {
                return false;
            }

            Predicate rhs = (Predicate)other;

            return new EqualsBuilder()
                .append(this.Name, rhs.Name)
                .append(this.Arguments, rhs.Arguments)
                .append(this.IsNegated, rhs.IsNegated)
                .isEquals();
        }

        @Override
        public int hashCode() {

            return new HashCodeBuilder(17, 31)
                    .append(this.Name)
                    .append(this.Arguments)
                    .append(this.IsNegated)
                    .toHashCode();

        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            if (this.IsNegated) {
                sb.append("~");
            }
            sb.append(this.Name);
            sb.append("(");
            for (int i = 0; i < this.Arguments.size(); ++i) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(this.Arguments.get(i));
            }
            sb.append(")");
            return sb.toString();
        }

    }

    public enum GroundingMode {
        AllPossible,
        NewAllPossible,
        ByExtension,
        AsRead
    }

    public static class Rule {

        private static class GroundingSource {

            public GroundingSource(int indexPredicate, int indexArgument) {
                this.IndexPredicate = indexPredicate;
                this.IndexArgument = indexArgument;
            }

            public final int IndexPredicate;
            public final int IndexArgument;

        }

        private static class ArgumentGroundingSources {

            public ArgumentGroundingSources(String name, ImmutableList<GroundingSource> groundingSources) {
                this.Name = name;
                this.GroundingSources = groundingSources;
            }

            public final String Name;
            public final ImmutableList<GroundingSource> GroundingSources;

        }

        public Rule(double weight, double power, ImmutableList<Predicate> head, ImmutableList<Predicate> body) {

            ImmutableList.Builder<Predicate> orderingRemovedBody = ImmutableList.builder();
            ImmutableList.Builder<ImmutableList<String>> orderings = ImmutableList.builder();
            HashMap<String, ImmutableList.Builder<GroundingSource>> groundingSources = new HashMap<>();

            for (Predicate predicate : head) {

                if (predicate.Name.equals("#NONSYMMETRIC")) {
                    throw new UnsupportedOperationException("Unexpected #NONSYMMETRIC keyword in head of rule");
                }

            }

            for (int indexPredicate = 0; indexPredicate < body.size(); ++indexPredicate) {

                Predicate predicate = body.get(indexPredicate);
                if (predicate.Name.equals("#NONSYMMETRIC")) {

                    if (predicate.IsNegated) {
                        throw new UnsupportedOperationException("Negation attached to #NONSYMMETRIC keyword");
                    }

                    orderings.add(predicate.Arguments);

                } else {

                    orderingRemovedBody.add(predicate);
                    for (int indexArgument = 0; indexArgument < predicate.Arguments.size(); ++indexArgument) {
                        ImmutableList.Builder<GroundingSource> argGroundingSource = groundingSources.getOrDefault(predicate.Arguments.get(indexArgument), null);
                        if (argGroundingSource == null) {
                            argGroundingSource = ImmutableList.builder();
                            groundingSources.put(predicate.Arguments.get(indexArgument), argGroundingSource);
                        }

                        argGroundingSource.add(new GroundingSource(indexPredicate, indexArgument));
                    }

                }

            }

            ImmutableList.Builder<ArgumentGroundingSources> groundingSourceBuilder = ImmutableList.builder();
            for (Map.Entry<String, ImmutableList.Builder<GroundingSource>> entry : groundingSources.entrySet()) {
                groundingSourceBuilder.add(new ArgumentGroundingSources(entry.getKey(), entry.getValue().build()));
            }

            this.Weight = weight;
            this.Power = power;
            this.Head = head;
            this.Body = orderingRemovedBody.build();
            this.Orderings = orderings.build();
            this.ArgumentGroundingSources = groundingSourceBuilder.build();

        }


        // DBC: Added this Rule to conveniently read in pre-grounded rules (which only have heads)
        public Rule(double weight, double power, ImmutableList<Predicate> head) {

            ImmutableList.Builder<Predicate> orderingRemovedBody = ImmutableList.builder();
            ImmutableList.Builder<ImmutableList<String>> orderings = ImmutableList.builder();
            HashMap<String, ImmutableList.Builder<GroundingSource>> groundingSources = new HashMap<>();

            for (Predicate predicate : head) {

                if (predicate.Name.equals("#NONSYMMETRIC")) {
                    throw new UnsupportedOperationException("Unexpected #NONSYMMETRIC keyword in head of rule");
                }

            }

            ImmutableList.Builder<ArgumentGroundingSources> groundingSourceBuilder = ImmutableList.builder();
            for (Map.Entry<String, ImmutableList.Builder<GroundingSource>> entry : groundingSources.entrySet()) {
                groundingSourceBuilder.add(new ArgumentGroundingSources(entry.getKey(), entry.getValue().build()));
            }

            this.Weight = weight;
            this.Power = power;
            this.Head = head;
            //this.Body = orderingRemovedBody.build();
            //this.Orderings = orderings.build();
            //this.ArgumentGroundingSources = groundingSourceBuilder.build();
            this.Body = null;
            this.Orderings = null;
            this.ArgumentGroundingSources = null;
        }



        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            if (Double.isNaN(this.Weight)) {
                sb.append("constraint");
            } else {
                sb.append(this.Weight);
            }
            sb.append("} ");
            for (int i = 0; i < this.Body.size(); ++i) {
                if (i > 0) {
                    sb.append(" & ");
                }
                sb.append(this.Body.get(i).toString());
            }
            sb.append(" >> ");
            for (int i = 0; i < this.Head.size(); ++i) {
                if (i > 0) {
                    sb.append(" | ");
                }
                sb.append(this.Head.get(i).toString());
            }
            if (!Double.isNaN(this.Weight)) {
                sb.append(" {");
                if (this.Power == 2) {
                    sb.append("squared");
                } else {
                    sb.append(this.Power);
                }
                sb.append("}");
            }
            return sb.toString();
        }

        public static void addGroundingsToBuilder(
            List<Rule> rules,
            ProbabilisticSoftLogicProblem.GroundedRuleHandler builder,
            ProbabilisticSoftLogicPredicateManager predicateManager,
            LogicManager<Integer, Double> logicManager,
            VariableType<Integer> variableType,
            GroundingMode groundingMode) {

            if (groundingMode == GroundingMode.AllPossible) {

                for (ProbabilisticSoftLogicProblem.Rule rule : rules) {

                    rule.addAllGroundingsToBuilder(builder, predicateManager);

                }
            } else if (groundingMode == GroundingMode.NewAllPossible) {
                for (ProbabilisticSoftLogicProblem.Rule rule : rules) {
                    rule.addAllGroundingsToBuilder(builder, logicManager, variableType);
                }
            } else {

                Rule.addGroundingsToBuilderByExtension(
                        rules,
                        builder,
                        predicateManager,
                        groundingMode != GroundingMode.AsRead);

            }

        }

        // allowPredicateCreation is a temporary measure
        // to restrict the groundings to exactly what we have read
        private static void addGroundingsToBuilderByExtension(
                List<Rule> rules,
                ProbabilisticSoftLogicProblem.GroundedRuleHandler builder,
                ProbabilisticSoftLogicPredicateManager predicateManager,
                boolean allowPredicateCreation ) {

            HashSet<Integer> newPredicates = new HashSet<>();
            List<HashSet<String>> groundingsAlreadyAdded = new ArrayList<>();
            for (int i = 0; i < rules.size(); ++i) {
                groundingsAlreadyAdded.add(new HashSet<>());
            }

            for (String predicateName : predicateManager.getPredicateNames()) {
                for (int id : predicateManager.getIdsForPredicateName(predicateName)) {
                    newPredicates.add(id);
                }
            }

            while(!newPredicates.isEmpty()) {

                HashSet<Integer> currentPredicates = newPredicates;
                newPredicates = new HashSet<>();

                for (int indexRule = 0; indexRule < rules.size(); ++indexRule) {
                    System.out.println("Current rule: " + indexRule);
                    for (int predicateId : currentPredicates) {

                        rules.get(indexRule).extendGroundingsAndAddToBuilder(
                                predicateManager.getPredicateFromId(predicateId),
                                builder,
                                predicateManager,
                                groundingsAlreadyAdded.get(indexRule),
                                newPredicates,
                                allowPredicateCreation );

                    }

                }

            }

        }

        private void extendGroundingsAndAddToBuilder(
                Predicate groundingExtension,
                ProbabilisticSoftLogicProblem.GroundedRuleHandler builder,
                ProbabilisticSoftLogicPredicateManager predicateManager,
                HashSet<String> groundingsAlreadyAdded,
                HashSet<Integer> newPredicates,
                boolean allowPredicateCreation ) {

            boolean[] bodyNegations = new boolean[this.Body.size()];
            for (int i = 0; i < this.Body.size(); ++i) {
                bodyNegations[i] = this.Body.get(i).IsNegated;
            }

            boolean[] headNegations = new boolean[this.Head.size()];
            for (int i = 0; i < this.Head.size(); ++i) {
                headNegations[i] = this.Head.get(i).IsNegated;
            }

            List<List<String>> allArgumentGroundings = new ArrayList<>();

            for (ArgumentGroundingSources argumentGroundingSource : this.ArgumentGroundingSources) {

                HashSet<String> argumentGroundings = new HashSet<>();
                for (GroundingSource source : argumentGroundingSource.GroundingSources) {
                    argumentGroundings.addAll(predicateManager.getArgumentGroundings(this.Body.get(source.IndexPredicate).Name, source.IndexArgument));
                }

                allArgumentGroundings.add(new ArrayList<>(argumentGroundings));

            }

            // iterate over possible places where we could plug in this predicate
            HashSet<String> visitedGroundings = new HashSet<>();
            for (int indexInsertion = 0; indexInsertion < this.Body.size(); ++indexInsertion) {

                // we can plug it in here
                if (this.Body.get(indexInsertion).Name.equals(groundingExtension.Name)
                        && this.Body.get(indexInsertion).Arguments.size() == groundingExtension.Arguments.size()) {

                    HashMap<String, String> extensionGroundings = new HashMap<>();
                    for (int indexExtensionArgument = 0; indexExtensionArgument < groundingExtension.Arguments.size(); ++indexExtensionArgument) {
                        extensionGroundings.put(this.Body.get(indexInsertion).Arguments.get(indexExtensionArgument), groundingExtension.Arguments.get(indexExtensionArgument));
                    }

                    String visitedKey = String.join("|", extensionGroundings.keySet());
                    if (!visitedGroundings.add(visitedKey)) {
                        continue;
                    }

                    List<List<String>> argumentValuesToTry = new ArrayList<>();
                    for (ArgumentGroundingSources sources : this.ArgumentGroundingSources) {
                        String extensionVal = extensionGroundings.getOrDefault(sources.Name, null);
                        if (extensionVal != null) {
                            argumentValuesToTry.add(Arrays.asList(extensionVal));
                        } else {
                            HashSet<String> availableValues = new HashSet<>();
                            for (GroundingSource source : sources.GroundingSources) {
                                availableValues.addAll(predicateManager.getArgumentGroundings(this.Body.get(source.IndexPredicate).Name, source.IndexArgument));
                            }
                            argumentValuesToTry.add(new ArrayList<>(availableValues));
                        }
                    }

                    CartesianProductIterator<String> argumentInstanceIterator = new CartesianProductIterator<>(argumentValuesToTry);
                    for (List<String> argumentInstance : argumentInstanceIterator) {

                        HashMap<String, String> argumentToGrounding = new HashMap<>();
                        StringBuilder groundingStringBuilder = new StringBuilder();
                        for (int indexArgument = 0; indexArgument < this.ArgumentGroundingSources.size(); ++indexArgument) {
                            argumentToGrounding.put(this.ArgumentGroundingSources.get(indexArgument).Name, argumentInstance.get(indexArgument));
                            if (indexArgument > 0) {
                                groundingStringBuilder.append(",");
                            }
                            groundingStringBuilder.append(this.ArgumentGroundingSources.get(indexArgument).Name);
                            groundingStringBuilder.append(":");
                            groundingStringBuilder.append(argumentInstance.get(indexArgument));
                        }

                        String groundingString = groundingStringBuilder.toString();

                        // don't add to this unless the grounding gets added;
                        // it could be the case that this grounding can't be added
                        // now but can be added later
                        if (groundingsAlreadyAdded.contains(groundingString)) {
                            continue;
                        }

                        if (!this.getIsGroundedOrderingAllowed(argumentToGrounding)) {
                            continue;
                        }

                        Map.Entry<int[], boolean[]> bodyIdResult = Rule.getPredicateIds(this.Body, argumentToGrounding, predicateManager, false);
                        // check whether the body (premise) can possibly hold
                        // only use this expansion if it can
                        double observedBodyConstant = 0;
                        boolean pruneGrounding = false;
                        for (int indexBody = 0; indexBody < this.Body.size(); ++indexBody) {
                            int bodyId = bodyIdResult.getKey()[indexBody];
                            double observedValue = bodyId < 0 ? (this.Body.get(indexBody).IsNegated ? 0 : 1) : predicateManager.getObservedWeight(bodyId);
                            if (!allowPredicateCreation && bodyId < 0) {
                                pruneGrounding = true;
                                break;
                            }
                            if (!Double.isNaN(observedValue) && bodyId >= 0) {
                                if (this.Body.get(indexBody).IsNegated)
                                    observedBodyConstant -= observedValue;
                                else
                                    observedBodyConstant += observedValue - 1;
                            }
                        }

                        if (pruneGrounding || observedBodyConstant + 1 <= 0)
                            continue;

                        // if these have -1, it is ok.  The builder will just treat these as observed
                        // with value 0
                        Map.Entry<int[], boolean[]> headIdResult = Rule.getPredicateIds(this.Head, argumentToGrounding, predicateManager, allowPredicateCreation);
                        for (int indexHead = 0; indexHead < this.Head.size(); ++indexHead) {
                            if (!allowPredicateCreation && headIdResult.getKey()[indexHead] < 0) {
                                pruneGrounding = true;
                                break;
                            }
                            if (headIdResult.getValue()[indexHead]) {
                                newPredicates.add(headIdResult.getKey()[indexHead]);
                            }
                        }

                        if (pruneGrounding) {
                            continue;
                        }

                        groundingsAlreadyAdded.add(groundingString);

                        // BUG BUGBUGBUG temporarily handle constraints by setting to high weight
                        if (Double.isNaN(this.Weight)) {
                            builder.addRule(headIdResult.getKey(), bodyIdResult.getKey(), headNegations, bodyNegations, 1, 1000);
                        } else {
                            builder.addRule(headIdResult.getKey(), bodyIdResult.getKey(), headNegations, bodyNegations, this.Power, this.Weight);
                        }

                    }
                }
            }

        }

        private void addAllGroundingsToBuilder(ProbabilisticSoftLogicProblem.GroundedRuleHandler builder,
                                               LogicManager<Integer, Double> logicManager,
                                               VariableType<Integer> variableType) {
            List<Formula<Integer>> disjunctionComponents = new ArrayList<>();
            boolean[] bodyNegations = new boolean[this.Body.size()];
            for (int i = 0; i < this.Body.size(); ++i) {
                bodyNegations[i] = this.Body.get(i).IsNegated;
                org.platanios.learn.logic.formula.Predicate<Integer> predicate = logicManager.getPredicate(this.Body.get(i).Name);
                List<Variable<Integer>> predicateArguments = this.Body.get(i).Arguments.stream().map(logicManager::getVariable).collect(Collectors.toList());
                if (this.Body.get(i).IsNegated)
                    disjunctionComponents.add(new Atom<>(predicate, predicateArguments));
                else
                    disjunctionComponents.add(new Negation<>(new Atom<>(predicate, predicateArguments)));
            }
            boolean[] headNegations = new boolean[this.Head.size()];
            for (int i = 0; i < this.Head.size(); ++i) {
                headNegations[i] = this.Head.get(i).IsNegated;
                org.platanios.learn.logic.formula.Predicate<Integer> predicate = logicManager.getPredicate(this.Head.get(i).Name);
                List<Variable<Integer>> predicateArguments = this.Head.get(i).Arguments.stream().map(logicManager::getVariable).collect(Collectors.toList());
                if (this.Head.get(i).IsNegated)
                    disjunctionComponents.add(new Negation<>(new Atom<>(predicate, predicateArguments)));
                else
                    disjunctionComponents.add(new Atom<>(predicate, predicateArguments));
            }
            Formula<Integer> ruleFormula = new Disjunction<>(disjunctionComponents);
            ExhaustiveGrounding<Integer, Double> exhaustiveGrounding = new ExhaustiveGrounding<>(logicManager);
            exhaustiveGrounding.ground(ruleFormula);
            List<List<GroundedPredicate<Integer, Double>>> predicateGroundings = exhaustiveGrounding.getGroundedPredicates();
            for (List<GroundedPredicate<Integer, Double>> groundedRulePredicates : predicateGroundings) {
                int[] bodyVariableIndexes = new int[this.Body.size()];
                for (int i = 0; i < this.Body.size(); ++i)
                    bodyVariableIndexes[i] = (int) groundedRulePredicates.get(i).getIdentifier();
                int[] headVariableIndexes = new int[this.Head.size()];
                for (int i = 0; i < this.Head.size(); ++i)
                    headVariableIndexes[i] = (int) groundedRulePredicates.get(this.Body.size() + i).getIdentifier();
                if (Double.isNaN(this.Weight)) {
                    builder.addRule(headVariableIndexes, bodyVariableIndexes, headNegations, bodyNegations, 1, 1000);
                } else {
                    builder.addRule(headVariableIndexes, bodyVariableIndexes, headNegations, bodyNegations, this.Power, this.Weight);
                }
            }
        }

        private void addAllGroundingsToBuilder(
                ProbabilisticSoftLogicProblem.GroundedRuleHandler builder,
                ProbabilisticSoftLogicPredicateManager predicateManager) {

//            // if this is a constraint, do nothing
//            if (Double.isNaN(this.Weight)) {
//                return;
//            }

            boolean[] bodyNegations = new boolean[this.Body.size()];
            for (int i = 0; i < this.Body.size(); ++i) {
                bodyNegations[i] = this.Body.get(i).IsNegated;
            }

            boolean[] headNegations = new boolean[this.Head.size()];
            for (int i = 0; i < this.Head.size(); ++i) {
                headNegations[i] = this.Head.get(i).IsNegated;
            }

            AbstractMap.SimpleEntry<List<String>, CartesianProductIterator<String>> allPossibleGroundings =
                    getAllPossibleGroundings(predicateManager);
            List<String> argumentNames = allPossibleGroundings.getKey();
            CartesianProductIterator<String>.IteratorState groundingIterator = allPossibleGroundings.getValue().iterator();

            HashMap<String, String> argumentToGrounding = new HashMap<>();

            while (groundingIterator.hasNext()) {
                List<String> groundings = groundingIterator.next();

                for (int i = 0; i < argumentNames.size(); ++i) {
                    argumentToGrounding.put(argumentNames.get(i), groundings.get(i));
                }

                if (!this.getIsGroundedOrderingAllowed(argumentToGrounding)
                        || !Rule.getIsGroundingCreationPossible(this.Body, argumentToGrounding, predicateManager)) {
                    continue;
                }

                Map.Entry<int[], boolean[]> bodyIdResult = getPredicateIds(this.Body, argumentToGrounding, predicateManager, true);
                Map.Entry<int[], boolean[]> headIdResult = getPredicateIds(this.Head, argumentToGrounding, predicateManager, true);

                // if we get this, it means we tried to instantiate a closed predicate to
                // a grounding which does not exist
                if (ArrayUtils.contains(bodyIdResult.getKey(), -1) || ArrayUtils.contains(headIdResult.getKey(), -1)) {
                    continue;
                }

                // BUG BUGBUGBUG temporarily handle constraints by setting to high weight
                if (Double.isNaN(this.Weight)) {
                    builder.addRule(headIdResult.getKey(), bodyIdResult.getKey(), headNegations, bodyNegations, 1, 1000);
                } else {
                    builder.addRule(headIdResult.getKey(), bodyIdResult.getKey(), headNegations, bodyNegations, this.Power, this.Weight);
                }

            }

        }

        private boolean getIsGroundedOrderingAllowed(
                HashMap<String, String> argumentToGrounding) {

            for (List<String> ordering : this.Orderings) {
                int maxSeenId = 0;
                for (String argument : ordering) {
                    try {
                        int idArgument = Integer.parseInt(argumentToGrounding.get(argument));
                        if (idArgument <= maxSeenId) {
                            return false;
                        }
                        maxSeenId = idArgument;
                    } catch (NumberFormatException e) {
                        throw new UnsupportedOperationException("ordering can only be imposed on numeric entities");
                    }
                }
            }

            return true;

        }

        private AbstractMap.SimpleEntry<List<String>, CartesianProductIterator<String>> getAllPossibleGroundings(
                ProbabilisticSoftLogicPredicateManager predicateManager) {

            HashSet<String> argumentGroundings = new HashSet<>();
            HashSet<String> argumentNames = new HashSet<>();

            // get all possible groundings for each of the named arguments in the rule
            for (Predicate predicate : Sets.union(new HashSet<>(this.Head), new HashSet<>(this.Body))) {
                for (int i = 0; i < predicate.Arguments.size(); ++i) {

                    Set<String> groundingsForPredicateAtPosition = predicateManager.getArgumentGroundings(predicate.Name, i);
                    argumentGroundings.addAll(groundingsForPredicateAtPosition);
                    argumentNames.add(predicate.Arguments.get(i));

                }
            }

            List<String> argumentNamesList = new ArrayList<>(argumentNames);
            ArrayList<List<String>> argumentGroundingValues = new ArrayList<>();
            for (int i = 0; i < argumentNamesList.size(); ++i) {
                argumentGroundingValues.add(new ArrayList<>(argumentGroundings));
            }
            CartesianProductIterator<String> groundingIterator = new CartesianProductIterator<>(argumentGroundingValues);
            return new AbstractMap.SimpleEntry<>(argumentNamesList, groundingIterator);

        }

        private static boolean getIsGroundingCreationPossible(
                List<Predicate> predicates,
                HashMap<String, String> groundings,
                ProbabilisticSoftLogicPredicateManager predicateManager) {

            for (int i = 0; i < predicates.size(); ++i) {
                Predicate lookup = Rule.createGroundedPredicate(predicates.get(i), groundings);
                int id = predicateManager.getIdForPredicate(lookup);
                if (id < 0 && predicateManager.getIsClosedPredicate(lookup.Name)) {
                    return false;
                }
            }

            return true;
        }

        private static Map.Entry<int[], boolean[]> getPredicateIds(
                List<Predicate> predicates,
                HashMap<String, String> groundings,
                ProbabilisticSoftLogicPredicateManager predicateManager,
                boolean allowPredicateCreation) {

            int[] result = new int[predicates.size()];
            boolean[] isAdded = new boolean[predicates.size()];

            for (int i = 0; i < predicates.size(); ++i) {
                Predicate lookup = Rule.createGroundedPredicate(predicates.get(i), groundings);
                if (predicateManager.getIsClosedPredicate(lookup.Name)) {
                    result[i] = predicateManager.getIdForPredicate(lookup);
                    isAdded[i] = false;
                } else {
                    if (allowPredicateCreation) {
                        Map.Entry<Boolean, Integer> getAddResult = predicateManager.getOrAddPredicate(lookup);
                        result[i] = getAddResult.getValue();
                        isAdded[i] = getAddResult.getKey();
                    } else {
                        result[i] = predicateManager.getIdForPredicate(lookup);
                        isAdded[i] = false;
                    }
                }
            }

            return new AbstractMap.SimpleEntry<>(result, isAdded);
        }

        private static Predicate createGroundedPredicate(Predicate template, HashMap<String, String> groundings) {
            ImmutableList.Builder<String> predicateGroundings = ImmutableList.builder();
            for (int j = 0; j < template.Arguments.size(); ++j) {
                predicateGroundings.add(groundings.get(template.Arguments.get(j)));
            }
            return new Predicate(template.Name, predicateGroundings.build(), false);
        }

        // NaN indicates constraint
        public final double Weight;
        public final double Power;
        public final ImmutableList<Predicate> Head;
        public final ImmutableList<Predicate> Body;
        public final ImmutableList<ImmutableList<String>> Orderings;
        public final ImmutableList<ArgumentGroundingSources> ArgumentGroundingSources;

    }

    public static abstract class GroundedRuleHandler {
        abstract GroundedRuleHandler addRule(
                int[] headVariableIndexes,
                int[] bodyVariableIndexes,
                boolean[] headNegations,
                boolean[] bodyNegations,
                double power,
                double weight);
    }


    // DBC: Added this class to use pre-grounded rules to generate a PSL program builder (based on ProblemSerializer)
    public static final class PregroundRuleHandler extends GroundedRuleHandler {

        private final Builder builder;

        private PregroundRuleHandler(Builder builder) {
            this.builder = builder;
        }

        public static Builder createBuilder(
                    List<Rule> rules,
                    ProbabilisticSoftLogicPredicateManager predicateManager,
                    LogicManager<Integer, Double> logicManager,
                    VariableType<Integer> variableType,
                    List<ProbabilisticSoftLogicProblem.Rule> groundRules) {

            ProbabilisticSoftLogicPredicateManager.IdWeights observedIdsAndWeights =
                    predicateManager.getAllObservedWeights();

            ProbabilisticSoftLogicProblem.Builder builder =
                    new ProbabilisticSoftLogicProblem.Builder(
                            observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, predicateManager.size() - observedIdsAndWeights.Ids.length);


            PregroundRuleHandler pregroundRuleHandler = new PregroundRuleHandler(builder);
            for (Rule rule : groundRules) {
                int [] headVariableIndexes = new int[rule.Head.size()];
                boolean [] headNegations= new boolean[rule.Head.size()];
                int i = 0;
                for (Predicate pred : rule.Head) {
                    // the pre-grounded rules will sometimes be negated; need to un-negate to look-up
                    Predicate unNegated = null;
                    if (pred.IsNegated) {
                        unNegated = new Predicate(pred.Name, pred.Arguments, false);
                    }
                    else {
                        unNegated = pred;
                    }

                    int temp = predicateManager.getIdForPredicate(unNegated);

                    headVariableIndexes[i] = temp;
                    headNegations[i] = pred.IsNegated;
                    i++;
                }
                pregroundRuleHandler.addRule(headVariableIndexes, new int[0], headNegations, new boolean[0], rule.Power, rule.Weight);
            }
            pregroundRuleHandler.addRule(new int[] {-1}, new int[] {-1}, new boolean[] {false}, new boolean[] {false}, Double.NaN, Double.NaN);

            return builder;

        }

        GroundedRuleHandler addRule(
                int[] headVariableIndexes,
                int[] bodyVariableIndexes,
                boolean[] headNegations,
                boolean[] bodyNegations,
                double power,
                double weight) {

            this.builder.addRule(headVariableIndexes, bodyVariableIndexes, headNegations, bodyNegations, power, weight);

            return this;

        }

    }



    public static final class ProblemSerializer extends GroundedRuleHandler {

        private final OutputStream outputStream;
        private final Builder builder;

        private ProblemSerializer(OutputStream outputStream, Builder builder) {
            this.outputStream = outputStream;
            this.builder = builder;
        }

        public static Builder write(
                OutputStream outputStream,
                List<Rule> rules,
                ProbabilisticSoftLogicPredicateManager predicateManager,
                LogicManager<Integer, Double> logicManager,
                VariableType<Integer> variableType,
                GroundingMode groundingMode) throws IOException {

            ObjectOutputStream objectOutputStream = new ObjectOutputStream(outputStream);
            objectOutputStream.writeObject(predicateManager);

            ProbabilisticSoftLogicPredicateManager.IdWeights observedIdsAndWeights =
                    predicateManager.getAllObservedWeights();
            ProbabilisticSoftLogicProblem.Builder builder =
                new ProbabilisticSoftLogicProblem.Builder(
                        observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, predicateManager.size() - observedIdsAndWeights.Ids.length);

            try {
                ProblemSerializer serializer = new ProblemSerializer(outputStream, builder);
                Rule.addGroundingsToBuilder(rules, serializer, predicateManager, logicManager, variableType, groundingMode);
                serializer.addRule(new int[] {-1}, new int[] {-1}, new boolean[] {false}, new boolean[] {false}, Double.NaN, Double.NaN);
            } catch (UnsupportedOperationException e){
                if (e.getMessage() != null && e.getMessage().equals("IOException while writing rule")) {
                    throw (IOException) e.getCause();
                }
            }

            return builder;

        }

        public static Map.Entry<ProbabilisticSoftLogicPredicateManager, ProbabilisticSoftLogicProblem.Builder> read(InputStream inputStream) throws IOException, ClassNotFoundException {

            ObjectInputStream objectInputStream = new ObjectInputStream(inputStream);
            ProbabilisticSoftLogicPredicateManager predicateManager = (ProbabilisticSoftLogicPredicateManager) objectInputStream.readObject();

            ProbabilisticSoftLogicPredicateManager.IdWeights observedIdsAndWeights =
                    predicateManager.getAllObservedWeights();
            ProbabilisticSoftLogicProblem.Builder builder =
                    new ProbabilisticSoftLogicProblem.Builder(
                            observedIdsAndWeights.Ids, observedIdsAndWeights.Weights, predicateManager.size() - observedIdsAndWeights.Ids.length);

            boolean shouldReadRule = true;
            while (shouldReadRule) {
                shouldReadRule = ProblemSerializer.readRuleAndAddToBuilder(builder, inputStream);
            }

            return new AbstractMap.SimpleEntry<>(predicateManager, builder);

        }

        GroundedRuleHandler addRule(
                 int[] headVariableIndexes,
                 int[] bodyVariableIndexes,
                 boolean[] headNegations,
                 boolean[] bodyNegations,
                 double power,
                 double weight) {

            this.builder.addRule(headVariableIndexes, bodyVariableIndexes, headNegations, bodyNegations, power, weight);

            try {
                UnsafeSerializationUtilities.writeInt(outputStream, headVariableIndexes.length);
                UnsafeSerializationUtilities.writeIntArray(outputStream, headVariableIndexes);
                UnsafeSerializationUtilities.writeInt(outputStream, bodyVariableIndexes.length);
                UnsafeSerializationUtilities.writeIntArray(outputStream, bodyVariableIndexes);
                UnsafeSerializationUtilities.writeInt(outputStream, headNegations.length);
                UnsafeSerializationUtilities.writeBooleanArray(outputStream, headNegations);
                UnsafeSerializationUtilities.writeInt(outputStream, bodyNegations.length);
                UnsafeSerializationUtilities.writeBooleanArray(outputStream, bodyNegations);
                UnsafeSerializationUtilities.writeDouble(outputStream, power);
                UnsafeSerializationUtilities.writeDouble(outputStream, weight);
            } catch (IOException e) {
                throw new UnsupportedOperationException("IOException while writing rule", e);
            }

            return this;

        }

        private static boolean readRuleAndAddToBuilder(Builder builder, InputStream inputStream) throws IOException {
            int headVariableIndexesLength = UnsafeSerializationUtilities.readInt(inputStream);
            int[] headVariableIndexes = UnsafeSerializationUtilities.readIntArray(inputStream, headVariableIndexesLength);
            int bodyVariableIndexesLength = UnsafeSerializationUtilities.readInt(inputStream);
            int[] bodyVariableIndexes = UnsafeSerializationUtilities.readIntArray(inputStream, bodyVariableIndexesLength);
            int headNegationsLength = UnsafeSerializationUtilities.readInt(inputStream);
            boolean[] headNegations = UnsafeSerializationUtilities.readBooleanArray(inputStream, headNegationsLength);
            int bodyNegationsLength = UnsafeSerializationUtilities.readInt(inputStream);
            boolean[] bodyNegations = UnsafeSerializationUtilities.readBooleanArray(inputStream, bodyNegationsLength);
            double power = UnsafeSerializationUtilities.readDouble(inputStream);
            double weight = UnsafeSerializationUtilities.readDouble(inputStream);

            // indicates that we have finished reading rules
            if (headVariableIndexes[0] == -1) {
                return false;
            }

            builder.addRule(headVariableIndexes, bodyVariableIndexes, headNegations, bodyNegations, power, weight);
            return true;
        }

    }

    public static final class Builder extends GroundedRuleHandler {
        private final BiMap<Integer, Integer> externalToInternalIndexesMapping;

        private final Map<Integer, Double> observedVariableValues;
        //private final HashMap<String, FunctionTerm> functionTerms = new HashMap<>();
        private final List<FunctionTerm> functionTerms = new ArrayList<>();
        private final List<Constraint> constraints = new ArrayList<>();
        private final Map<Integer, List<Integer>> externalPredicateIdToTerms;

        private int nextInternalIndex = 0;

        public Builder(
                int[] observedVariableIndexes,
                double[] observedVariableValues,
                int numberOfUnobservedVariables) {

            ImmutableMap.Builder<Integer, Double> observedVariableValueBuilder = ImmutableMap.builder();
            if ((observedVariableIndexes == null) != (observedVariableValues == null)) {
                throw new IllegalArgumentException(
                        "The provided indexes for the observed variables must much the corresponding provided values."
                );
            }
            if (observedVariableIndexes != null) {
                if (observedVariableIndexes.length != observedVariableValues.length) {
                    throw new IllegalArgumentException(
                            "The provided indexes array for the observed variables must " +
                                    "have the same length the corresponding provided values array."
                    );
                }
                for (int i = 0; i < observedVariableIndexes.length; ++i) {
                    observedVariableValueBuilder.put(observedVariableIndexes[i], observedVariableValues[i]);
                }
            }

            // special case - always add -1 as an Id with observed value 0.
            // predicates which are grounded outside of a closed set will thus have value 0
            observedVariableValueBuilder.put(-1, 0.0);
            this.observedVariableValues = observedVariableValueBuilder.build();
            this.externalToInternalIndexesMapping = HashBiMap.create(numberOfUnobservedVariables);
            this.externalPredicateIdToTerms = new HashMap<>(numberOfUnobservedVariables + observedVariableIndexes.length, 1);
        }

        public int getNumberOfTerms() { return this.functionTerms.size(); }

        @Override
        Builder addRule(
                int[] headVariableIndexes,
                int[] bodyVariableIndexes,
                boolean[] headNegations,
                boolean[] bodyNegations,
                double power,
                double weight) {
            RulePart headPart = convertRulePartToInternalRepresentation(headVariableIndexes, headNegations, true);
            RulePart bodyPart = convertRulePartToInternalRepresentation(bodyVariableIndexes, bodyNegations, false);
            double ruleMaximumValue = 1 + headPart.observedConstant + bodyPart.observedConstant;
            if (ruleMaximumValue <= 0)
                return this;
            int[] variableIndexes = Utilities.union(headPart.variableIndexes, bodyPart.variableIndexes);
            if (variableIndexes.length == 0)
                return this;
            int indexTerm = this.functionTerms.size();
            LinearFunction linearFunction = new LinearFunction(Vectors.dense(variableIndexes.length), ruleMaximumValue);
            for (int headVariable = 0; headVariable < headPart.variableIndexes.length; headVariable++) {
                Vector coefficients = Vectors.dense(variableIndexes.length);
                if (headPart.negations[headVariable]) {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, headPart.variableIndexes[headVariable]), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, headPart.variableIndexes[headVariable]), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                }
            }
            for (int bodyVariable = 0; bodyVariable < bodyPart.variableIndexes.length; bodyVariable++) {
                List<Integer> predicateTermIndices = this.externalPredicateIdToTerms.getOrDefault(bodyPart.variableIndexes[bodyVariable], null);
                if (predicateTermIndices == null) {
                    predicateTermIndices = new ArrayList<>(200);
                    this.externalPredicateIdToTerms.put(bodyPart.variableIndexes[bodyVariable], predicateTermIndices);
                }
                predicateTermIndices.add(indexTerm);
                Vector coefficients = Vectors.dense(variableIndexes.length);
                if (bodyPart.negations[bodyVariable]) {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, bodyPart.variableIndexes[bodyVariable]), -1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, 0));
                } else {
                    coefficients.set(ArrayUtils.indexOf(variableIndexes, bodyPart.variableIndexes[bodyVariable]), 1);
                    linearFunction = linearFunction.add(new LinearFunction(coefficients, -1));
                }
            }

            for (int headVariable = 0; headVariable < headVariableIndexes.length; ++headVariable) {
                List<Integer> predicateTermIndices = this.externalPredicateIdToTerms.getOrDefault(headVariableIndexes[headVariable], null);
                if (predicateTermIndices == null) {
                    predicateTermIndices = new ArrayList<>(200);
                    this.externalPredicateIdToTerms.put(headVariableIndexes[headVariable], predicateTermIndices);
                }
                predicateTermIndices.add(indexTerm);
            }
            for (int bodyVariable = 0; bodyVariable < bodyVariableIndexes.length; ++bodyVariable) {
                List<Integer> predicateTermIndices = this.externalPredicateIdToTerms.getOrDefault(bodyVariableIndexes[bodyVariable], null);
                if (predicateTermIndices == null) {
                    predicateTermIndices = new ArrayList<>(200);
                    this.externalPredicateIdToTerms.put(bodyVariableIndexes[bodyVariable], predicateTermIndices);
                }
                predicateTermIndices.add(indexTerm);
            }

            FunctionTerm term = new FunctionTerm(variableIndexes, linearFunction, weight, power);
            // functionTerms.putIfAbsent(term.toString(), term);
            functionTerms.add(term);
            return this;
        }

        // BUG BUGBUGBUG: we should handle not adding redundant rules since we handle that on rules
        public Builder addConstraint(AbstractConstraint constraint, int... externalVariableIndexes) {
            List<Integer> internalVariableIndexes = new ArrayList<>();
            for (int externalVariableIndex : externalVariableIndexes) {
                int internalVariableIndex = externalToInternalIndexesMapping.getOrDefault(externalVariableIndex, -1);
                if (internalVariableIndex < 0) {
                    internalVariableIndex = nextInternalIndex++;
                    externalToInternalIndexesMapping.put(externalVariableIndex, internalVariableIndex);
                }
                internalVariableIndexes.add(internalVariableIndex);
            }
            constraints.add(new Constraint(constraint, Ints.toArray(internalVariableIndexes)));
            return this;
        }

        public ProbabilisticSoftLogicProblem build() {
            return new ProbabilisticSoftLogicProblem(this);
        }

        private RulePart convertRulePartToInternalRepresentation(int[] externalVariableIndexes,
                                                                 boolean[] negations,
                                                                 boolean isRuleHeadVariable) {
            List<Integer> internalVariableIndexes = new ArrayList<>();
            List<Boolean> internalVariableNegations = new ArrayList<>();
            double observedConstant = 0;
            for (int i = 0; i < externalVariableIndexes.length; ++i) {
                double observedValue = observedVariableValues.getOrDefault(externalVariableIndexes[i], Double.NaN);
                if (!Double.isNaN(observedValue)) {
                    if (isRuleHeadVariable == negations[i])
                        observedConstant += observedValue - 1;
                    else
                        observedConstant -= observedValue;
                } else {
                    int internalVariableIndex =
                            externalToInternalIndexesMapping.getOrDefault(externalVariableIndexes[i], -1);
                    if (internalVariableIndex < 0) {
                        internalVariableIndex = nextInternalIndex++;
                        externalToInternalIndexesMapping.put(externalVariableIndexes[i], internalVariableIndex);
                    }
                    internalVariableIndexes.add(internalVariableIndex);
                    internalVariableNegations.add(negations[i]);
                }
            }
            return new RulePart(
                    Ints.toArray(internalVariableIndexes),
                    Booleans.toArray(internalVariableNegations),
                    observedConstant
            );
        }

        private static class RulePart {
            private final int[] variableIndexes;
            private final boolean[] negations;
            private final double observedConstant;

            private RulePart(int[] variableIndexes,
                            boolean[] negations,
                            double observedConstant) {
                this.variableIndexes = variableIndexes;
                this.negations = negations;
                this.observedConstant = observedConstant;
            }
        }

        private static class FunctionTerm {

            private final LinearFunction linearFunction;
            private final int[] variableIndexes;
            private final double power;
            private final double weight;

            private FunctionTerm(int[] variableIndexes,
                                LinearFunction linearFunction,
                                double weight,
                                double power) {
                this.linearFunction = linearFunction;
                this.variableIndexes = variableIndexes;
                this.power = power;
                this.weight = weight;
            }

            @Override
            public String toString() {
                StringBuilder sb = new StringBuilder();
                sb.append("linFunc_a:[");
                Vector a = this.linearFunction.getA();
                for (int iComp = 0; iComp < a.size(); ++iComp) {
                    if (iComp > 0) {
                        sb.append(",");
                    }
                    sb.append(a.get(iComp));
                }
                sb.append("];linFunc_b:");
                sb.append(this.linearFunction.getB());
                sb.append(";idx:");
                for (int iIdx = 0; iIdx < this.variableIndexes.length; ++iIdx) {
                    if (iIdx > 0) {
                        sb.append(",");
                    }
                    sb.append(this.variableIndexes[iIdx]);
                }
                sb.append(";pwr:");
                sb.append(this.power);
                return sb.toString();
            }
        }
    }

    private ProbabilisticSoftLogicProblem(Builder builder) {
        this.externalToInternalIndexesMapping = builder.externalToInternalIndexesMapping;
        SumFunction.Builder sumFunctionBuilder = new SumFunction.Builder(externalToInternalIndexesMapping.size());
        for (Builder.FunctionTerm function : builder.functionTerms) {
        // for (Builder.FunctionTerm function : builder.functionTerms.values()) {
            MaxFunction.Builder maxFunctionBuilder = new MaxFunction.Builder(externalToInternalIndexesMapping.size());
            maxFunctionBuilder.addConstantTerm(0);
            maxFunctionBuilder.addFunctionTerm(function.linearFunction);
            sumFunctionBuilder.addTerm(
                    new ProbabilisticSoftLogicSumFunctionTerm(
                            maxFunctionBuilder.build(),
                            function.power,
                            function.weight),
                    function.variableIndexes
            );
        }
        this.objectiveFunction = new ProbabilisticSoftLogicFunction(sumFunctionBuilder);
        this.constraints = ImmutableSet.copyOf(builder.constraints);

        this.externalPredicateIdToTerms = builder.externalPredicateIdToTerms;

        for (int subProblemIndex = 0; subProblemIndex < objectiveFunction.getNumberOfTerms(); subProblemIndex++) {
            ProbabilisticSoftLogicSumFunctionTerm objectiveTerm =
                    (ProbabilisticSoftLogicSumFunctionTerm) objectiveFunction.getTerm(subProblemIndex);
            Vector coefficients = objectiveTerm.getLinearFunction().getA();
            if (objectiveTerm.getPower() == 2 && coefficients.size() > 2)
                subProblemCholeskyFactors.put(
                        subProblemIndex,
                        new CholeskyDecomposition(coefficients
                                                          .outer(coefficients)
                                                          .multiply(2 * objectiveTerm.weight)
                                                          .add(Matrix.generateIdentityMatrix(coefficients.size())))
                );
        }
    }

    public Map<Integer, Double> solve() {
        return this.solve(ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod.ALL, null, -1);
    }

    public Map<Integer, Double> solve(
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelectionMethod subProblemSelectionMethod,
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelector subProblemSelector,
        int numberOfSubProblemSamples) {
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder solverBuilder =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(
                        objectiveFunction,
                        Vectors.dense(objectiveFunction.getNumberOfVariables())
                )
                        .subProblemSolver((subProblem) -> solveProbabilisticSoftLogicSubProblem(subProblem, subProblemCholeskyFactors))
                        .subProblemSelector(subProblemSelector)
                        .subProblemSelectionMethod(subProblemSelectionMethod) // if this is not CUSTOM, it will override the subProblemSelector
                        .numberOfSubProblemSamples(numberOfSubProblemSamples)
                        .penaltyParameter(1)
                        .penaltyParameterSettingMethod(ConsensusAlternatingDirectionsMethodOfMultipliersSolver.PenaltyParameterSettingMethod.CONSTANT)
                        .checkForPointConvergence(false)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .logObjectiveValue(false)
                        .logGradientNorm(false)
                        .loggingLevel(3);
        for (Constraint constraint : constraints)
            solverBuilder.addConstraint(constraint.constraint, constraint.variableIndexes);
        ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver = solverBuilder.build();
        Vector solverResult = solver.solve();
        Map<Integer, Double> inferredValues = new HashMap<>(solverResult.size());
        for (int internalVariableIndex = 0; internalVariableIndex < solverResult.size(); internalVariableIndex++)
            inferredValues.put(externalToInternalIndexesMapping.inverse().get(internalVariableIndex),
                    solverResult.get(internalVariableIndex));
        return inferredValues;
    }

    private static void solveProbabilisticSoftLogicSubProblem(
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblem subProblem,
            Map<Integer, CholeskyDecomposition> subProblemCholeskyFactors
    ) {
        ProbabilisticSoftLogicSumFunctionTerm objectiveTerm =
                (ProbabilisticSoftLogicSumFunctionTerm) subProblem.objectiveTerm;
        subProblem.variables.set(
                subProblem.consensusVariables.sub(subProblem.multipliers.div(subProblem.penaltyParameter))
        );
        if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) > 0) {
            if (objectiveTerm.getPower() == 1) {
                subProblem.variables.subInPlace(objectiveTerm.getLinearFunction().getA()
                                                        .mult(objectiveTerm.getWeight() / subProblem.penaltyParameter));
            } else if (objectiveTerm.getPower() == 2) {
                double weight = objectiveTerm.getWeight();
                double constant = objectiveTerm.getLinearFunction().getB();
                subProblem.variables
                        .multInPlace(subProblem.penaltyParameter)
                        .subInPlace(objectiveTerm.getLinearFunction().getA()
                                            .mult(2 * weight * constant));
                if (subProblem.variables.size() == 1) {
                    double coefficient = objectiveTerm.getLinearFunction().getA().get(0);
                    subProblem.variables.divInPlace(2 * weight * coefficient * coefficient
                                                            + subProblem.penaltyParameter);
                } else if (subProblem.variables.size() == 2) {
                    double coefficient0 = objectiveTerm.getLinearFunction().getA().get(0);
                    double coefficient1 = objectiveTerm.getLinearFunction().getA().get(1);
                    double a0 = 2 * weight * coefficient0 * coefficient0 + subProblem.penaltyParameter;
                    double b1 = 2 * weight * coefficient1 * coefficient1 + subProblem.penaltyParameter;
                    double a1b0 = 2 * weight * coefficient0 * coefficient1;
                    subProblem.variables.set(
                            1,
                            (subProblem.variables.get(1) - a1b0 * subProblem.variables.get(0) / a0)
                                    / (b1 - a1b0 * a1b0 / a0)
                    );
                    subProblem.variables.set(
                            0,
                            (subProblem.variables.get(0) - a1b0 * subProblem.variables.get(1)) / a0
                    );
                } else {
                    try {
                        subProblem.variables.set(subProblemCholeskyFactors.get(subProblem.subProblemIndex).solve(subProblem.variables));
                    } catch (NonSymmetricMatrixException|NonPositiveDefiniteMatrixException e) {
                        System.err.println("Non-positive definite matrix!!!");
                    }
                }
            } else {
                subProblem.variables.set(
                        new NewtonSolver.Builder(
                                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemObjectiveFunction(
                                        objectiveTerm.getSubProblemObjectiveFunction(),
                                        subProblem.consensusVariables,
                                        subProblem.multipliers,
                                        subProblem.penaltyParameter
                                ),
                                subProblem.variables)
                                .lineSearch(new NoLineSearch(1))
                                .maximumNumberOfIterations(1)
                                .build()
                                .solve()
                );
            }
            if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) < 0) {
                subProblem.variables.set(
                        objectiveTerm.getLinearFunction().projectToHyperplane(subProblem.consensusVariables)
                );
            }
        }
    }

    public Map<Integer, Integer> getExternalToInternalIds() {
        return Collections.unmodifiableMap(this.externalToInternalIndexesMapping);
    }

    public Map<Integer, Integer> getInternalToExternalIds() {
        return Collections.unmodifiableMap(this.externalToInternalIndexesMapping.inverse());
    }

    public Map<Integer, List<Integer>> getExternalPredicateIdsToTerms() {
        return Collections.unmodifiableMap(this.externalPredicateIdToTerms);
    }

    public RandomWalkSampler.TermPredicateIdGetter getTermPredicateIdGetter() {
        return new RandomWalkSampler.TermPredicateIdGetter() {
            @Override
            public int[] getInternalPredicateIds(int term) {
                return ProbabilisticSoftLogicProblem.this.objectiveFunction.getTermIndices(term);
            }
        };
    }

    private static final class ProbabilisticSoftLogicFunction extends SumFunction {
        private ProbabilisticSoftLogicFunction(SumFunction.Builder sumFunctionBuilder) {
            super(sumFunctionBuilder);
        }

        // dangerous, should return unmodifiable collection
        public int[] getTermIndices(int term) {
            return this.termsVariables.get(term);
        }

        public LinearFunction getTermLinearFunction(int term) {
            return ((ProbabilisticSoftLogicSumFunctionTerm) terms.get(term)).getLinearFunction();
        }

        public double getTermPower(int term) {
            return ((ProbabilisticSoftLogicSumFunctionTerm) terms.get(term)).getPower();
        }

        public double getTermWeight(int term) {
            return ((ProbabilisticSoftLogicSumFunctionTerm) terms.get(term)).getWeight();
        }

    }

    private static final class ProbabilisticSoftLogicSubProblemObjectiveFunction extends AbstractFunction {
        private final LinearFunction linearFunction;
        private final double power;
        private final double weight;

        private ProbabilisticSoftLogicSubProblemObjectiveFunction(LinearFunction linearFunction,
                                                                  double power,
                                                                  double weight) {
            this.linearFunction = linearFunction;
            this.power = power;
            this.weight = weight;

        }

        @Override
        public boolean equals(Object other) {

            if (other == this) {
                return true;
            }

            if (!super.equals(other)) {
                return false;
            }

            if (!(other instanceof ProbabilisticSoftLogicSubProblemObjectiveFunction)) {
                return false;
            }

            ProbabilisticSoftLogicSubProblemObjectiveFunction rhs = (ProbabilisticSoftLogicSubProblemObjectiveFunction)other;

            return new EqualsBuilder()
                    .append(this.linearFunction, rhs.linearFunction)
                    .append(this.power, rhs.power)
                    .append(this.weight, rhs.weight)
                    .isEquals();

        }

        @Override
        public int hashCode() {

            return new HashCodeBuilder(53, 31)
                    .append(super.hashCode())
                    .append(this.linearFunction)
                    .append(this.power)
                    .append(this.weight)
                    .toHashCode();

        }

        @Override
        public double computeValue(Vector point) {
            return weight * Math.pow(linearFunction.computeValue(point), power);
        }

        @Override
        public Vector computeGradient(Vector point) {
            if (power > 0) {
                return linearFunction.computeGradient(point).mult(
                        weight * power * Math.pow(linearFunction.computeValue(point), power - 1)
                );
            } else {
                return Vectors.build(point.size(), point.type());
            }
        }

        @Override
        public Matrix computeHessian(Vector point) {
            if (power > 1) {
                Vector a = linearFunction.computeGradient(point);
                return a.outer(a).multiply(
                        weight * power * (power - 1) * Math.pow(linearFunction.computeValue(point), power - 2)
                );
            } else {
                return new Matrix(point.size(), point.size());
            }
        }
    }

    private static final class ProbabilisticSoftLogicSumFunctionTerm extends AbstractFunction {
        private final MaxFunction maxFunction;
        private final double power;
        private final double weight;

        private ProbabilisticSoftLogicSumFunctionTerm(MaxFunction maxFunction, double power, double weight) {
            this.maxFunction = maxFunction;
            this.power = power;
            this.weight = weight;
        }

        @Override
        public boolean equals(Object other) {

            if (other == this) {
                return true;
            }

            if (!super.equals(other)) {
                return false;
            }

            if (!(other instanceof ProbabilisticSoftLogicSumFunctionTerm)) {
                return false;
            }

            ProbabilisticSoftLogicSumFunctionTerm rhs = (ProbabilisticSoftLogicSumFunctionTerm)other;

            return new EqualsBuilder()
                    .append(this.maxFunction, rhs.maxFunction)
                    .append(this.power, rhs.power)
                    .append(this.weight, rhs.weight)
                    .isEquals();

        }

        @Override
        public int hashCode() {

            return new HashCodeBuilder(67, 17)
                    .append(super.hashCode())
                    .append(this.maxFunction)
                    .append(this.power)
                    .append(this.weight)
                    .toHashCode();

        }

        @Override
        public double computeValue(Vector point) {
            return weight * Math.pow(maxFunction.getValue(point), power);
        }

        private MaxFunction getMaxFunction() {
            return maxFunction;
        }

        private LinearFunction getLinearFunction() {
            return (LinearFunction) maxFunction.getFunctionTerm(0);
        }

        private double getPower() {
            return power;
        }

        private double getWeight() {
            return weight;
        }

        private ProbabilisticSoftLogicSubProblemObjectiveFunction getSubProblemObjectiveFunction() {
            return new ProbabilisticSoftLogicSubProblemObjectiveFunction(
                    (LinearFunction) maxFunction.getFunctionTerm(0),
                    power,
                    weight
            );
        }
    }

    private static final class Constraint {
        private final AbstractConstraint constraint;
        private final int[] variableIndexes;

        private Constraint(AbstractConstraint constraint, int[] variableIndexes) {
            this.constraint = constraint;
            this.variableIndexes = variableIndexes;
        }

        @Override
        public boolean equals(Object other) {
            if(!(other instanceof Constraint)) {
                return false;
            }

            if (other == this) {
                return true;
            }

            Constraint rhs = (Constraint) other;
            return new EqualsBuilder()
                    .append(this.constraint, rhs.constraint)
                    .append(this.variableIndexes, rhs.variableIndexes)
                    .isEquals();
        }

        @Override
        public int hashCode() {
            return new HashCodeBuilder(13, 37)
                    .append(this.constraint)
                    .append(this.variableIndexes)
                    .toHashCode();
        }

    }
}
