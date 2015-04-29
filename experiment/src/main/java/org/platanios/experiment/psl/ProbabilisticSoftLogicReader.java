package org.platanios.experiment.psl;

import org.platanios.experiment.psl.parser.ComplexPredicateParser;
import org.platanios.experiment.psl.parser.PrattParserExpression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.zip.DataFormatException;

/**
 * Created by Dan Schwartz 4/27/15
 * Reads files associated with probabilistic soft logic
 */
public class ProbabilisticSoftLogicReader {

    private ProbabilisticSoftLogicReader() {}

    public static class Predicate {

        public Predicate(String name, List<String> arguments, boolean isNegated) {
            this.Name = name;
            this.Arguments = arguments;
            this.IsNegated = isNegated;
        }

        public final String Name;
        public final List<String> Arguments;
        public final boolean IsNegated;

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

    public static class Rule {

        public Rule(double weight, double power, List<Predicate> head, List<Predicate> body) {

            this.Weight = weight;
            this.Power = power;
            this.Head = head;
            this.Body = body;

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

        public void addGroundingsToBuilder(
                ProbabilisticSoftLogicProblem.Builder builder,
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
            CartesianProductIterator<String> groundingIterator = allPossibleGroundings.getValue();

            HashMap<String, String> argumentToGrounding = new HashMap<>();

            for (List<String> groundings : groundingIterator) {

                for (int i = 0; i < argumentNames.size(); ++i) {
                    argumentToGrounding.put(argumentNames.get(i), groundings.get(i));
                }

                int[] bodyIds = getPredicateIds(this.Body, argumentToGrounding, predicateManager);
                int[] headIds = getPredicateIds(this.Head, argumentToGrounding, predicateManager);

                // BUG BUGBUGBUG temporarily handle constraints by setting to high weight
                if (Double.isNaN(this.Weight)) {
                    builder.addRule(headIds, bodyIds, headNegations, bodyNegations, 1, 1000);
                } else {
                    builder.addRule(headIds, bodyIds, headNegations, bodyNegations, this.Power, this.Weight);
                }

            }

        }

        private AbstractMap.SimpleEntry<List<String>, CartesianProductIterator<String>> getAllPossibleGroundings(
                ProbabilisticSoftLogicPredicateManager predicateManager) {

            HashMap<String, HashSet<String>> argumentGroundings = new HashMap<>();

            // get all possible groundings for each of the named arguments in the rule
            for (Predicate predicate : this.Head) {
                for (int i = 0; i < predicate.Arguments.size(); ++i) {

                    HashSet<String> groundingsForName = argumentGroundings.getOrDefault(predicate.Arguments.get(i), null);
                    if (groundingsForName == null) {
                        groundingsForName = new HashSet<>();
                        argumentGroundings.put(predicate.Arguments.get(i), groundingsForName);
                    }

                    Iterator<String> groundingIterator = predicateManager.getArgumentGroundings(predicate.Name, i);

                    while (groundingIterator.hasNext()) {

                        groundingsForName.add(groundingIterator.next());

                    }

                }
            }

            List<String> argumentNames = new ArrayList<>(argumentGroundings.keySet());
            ArrayList<List<String>> argumentGroundingValues = new ArrayList<>();
            for (String argumentName : argumentNames) {
                argumentGroundingValues.add(new ArrayList<>(argumentGroundings.get(argumentName)));
            }
            CartesianProductIterator<String> groundingIterator = new CartesianProductIterator<>(argumentGroundingValues);
            return new AbstractMap.SimpleEntry<>(argumentNames, groundingIterator);

        }

        private static int[] getPredicateIds(
                List<Predicate> predicates,
                HashMap<String, String> groundings,
                ProbabilisticSoftLogicPredicateManager predicateManager) {

            int[] result = new int[predicates.size()];
            for (int i = 0; i < predicates.size(); ++i) {
                ArrayList<String> predicateGroundings = new ArrayList<>();
                for (int j = 0; j < predicates.get(i).Arguments.size(); ++j) {
                    predicateGroundings.add(groundings.get(predicates.get(i).Arguments.get(j)));
                }
                Predicate lookup = new Predicate(predicates.get(i).Name, predicateGroundings, false);
                result[i] = predicateManager.getIdForPredicate(lookup);
                if (result[i] < 0) {
                    throw new UnsupportedOperationException("Unable to find predicate with current grounding.  Code issue.");
                }
            }

            return result;
        }

        // NaN indicates constraint
        public final double Weight;
        public final double Power;
        public final List<Predicate> Head;
        public final List<Predicate> Body;

    }

    public static List<Integer> readGroundingsAndAddToManager(
            ProbabilisticSoftLogicPredicateManager predicateManager,
            String predicateName,
            String filename) throws DataFormatException, IOException {

        BufferedReader br = null;

        try {

            File file = new File(filename);

            br = new BufferedReader(new FileReader(file));
            return readGroundingsAndAddToManager(predicateManager, predicateName, br);

        } finally {

            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        }

    }

    public static List<Integer> readGroundingsAndAddToManager(
            ProbabilisticSoftLogicPredicateManager predicateManager,
            String predicateName,
            BufferedReader reader) throws DataFormatException, IOException {

        String line;

        int lineNumber = 0;
        boolean hasWeights = false;
        while ((line = reader.readLine()) != null) {

            String[] lineFields = line.split("\t");
            if (lineNumber == 0 && lineFields.length == 3) {
                hasWeights = true;
            }

            double weight = 1;
            if (hasWeights) {
                if (lineFields.length != 3) {
                    throw new DataFormatException("If any line has a weight, all lines must have a weight.  Bad format on line: " + (lineNumber + 1));
                }
                weight = Double.parseDouble(lineFields[2]);
            } else if (lineFields.length != 2) {
                throw new DataFormatException("Bad format on line: " + (lineNumber + 1));
            }

            ArrayList<String> currentGrounding = new ArrayList<>();
            for (int indexEntity = 0; indexEntity < 2; ++indexEntity) {
                String entity = lineFields[indexEntity];
                entity = entity.trim();
                if (entity.isEmpty()) {
                    throw new DataFormatException("Empty entity on line: " + (lineNumber + 1));
                }
                currentGrounding.add(entity);
            }

            Predicate groundedPredicate = new Predicate(predicateName, currentGrounding, false);
            predicateManager.getOrAddPredicate(groundedPredicate, weight);

        }

        return predicateManager.getIdsForPredicateName(predicateName);

    }

    public static ArrayList<Rule> readRules(String filename) throws DataFormatException, IOException {

        BufferedReader br = null;

        try {

            File file = new File(filename);

            br = new BufferedReader(new FileReader(file));
            return readRules(br);

        } finally {

            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        }

    }

    public static ArrayList<Rule> readRules(BufferedReader reader) throws DataFormatException, IOException {

        ArrayList<Rule> result = new ArrayList<>();

        String line;

        int lineNumber = 0;
        while ((line = reader.readLine()) != null) {

            line = line.trim();

            if (!line.startsWith("{")) {
                throw new DataFormatException("Expected rules of the form {weight} body >> head : no leading curly brace found at line " + (lineNumber + 1));
            }

            int indexEndCurly = line.indexOf("}");
            if (indexEndCurly < 0) {
                throw new DataFormatException("Expected rules of the form {weight} body >> head : no closing curly brace found at line " + (lineNumber + 1));
            }

            String weightOrConstraint = line.substring(1, indexEndCurly);
            double weight = Double.NaN;
            if (!weightOrConstraint.equals("constraint")) {
                weight = Double.parseDouble(weightOrConstraint);
            }

            double power = Double.isNaN(weight) ? Double.NaN : 1;
            int lastOpenCurly = line.lastIndexOf('{');
            if (lastOpenCurly >= 0 && lastOpenCurly != 0) {
                int lastEndCurly = line.lastIndexOf('}');
                if (lastEndCurly != line.length() - 1) {
                    throw new DataFormatException("Expected closing curly brace for power indicator");
                }
                String powerString = line.substring(lastOpenCurly + 1, lastEndCurly);
                line = line.substring(0, lastOpenCurly);
                if (powerString.equals("squared")) {
                    power = 2;
                } else {
                    power = Double.parseDouble(powerString);
                }
            }

            String ruleString = line.substring(indexEndCurly + 1);
            String[] headBodyArr = ruleString.split(">>");
            if (headBodyArr.length != 2) {
                throw new DataFormatException("Expected rules of the form {weight} body >> head : number of head/body parts is wrong at line " + (lineNumber + 1));
            }

            PrattParserExpression headExpression = ComplexPredicateParser.parseRule(headBodyArr[1]);
            PrattParserExpression bodyExpression = ComplexPredicateParser.parseRule(headBodyArr[0]);

            PredicateTemplateOperatorList flattenedHead = flattenLogicExpression(headExpression);
            for (ComplexPredicateParser.OperatorType operator : flattenedHead.Operators) {
                if (operator != ComplexPredicateParser.OperatorType.DISJUNCTION) {
                    throw new UnsupportedOperationException("Only disjunctions are allowed in the head of a rule");
                }
            }

            PredicateTemplateOperatorList flattenedBody = flattenLogicExpression(bodyExpression);
            for (ComplexPredicateParser.OperatorType operator : flattenedBody.Operators) {
                if (operator != ComplexPredicateParser.OperatorType.CONJUNCTION) {
                    throw new UnsupportedOperationException("Only conjunctions are allowed in the body of a rule");
                }
            }

            Rule rule = new Rule(weight, power, flattenedHead.Predicates, flattenedBody.Predicates);
            result.add(rule);

            ++lineNumber;

        }

        return result;

    }

    private static class PredicateTemplateOperatorList {

        public final ArrayList<ComplexPredicateParser.OperatorType> Operators = new ArrayList<>();
        public final ArrayList<Predicate> Predicates = new ArrayList<>();

    }

    private static PredicateTemplateOperatorList flattenLogicExpression(PrattParserExpression expression) {

        if (ComplexPredicateParser.InfixBinaryOperatorExpression.class.isInstance(expression)) {

            ComplexPredicateParser.InfixBinaryOperatorExpression infixExpression = (ComplexPredicateParser.InfixBinaryOperatorExpression) expression;
            PredicateTemplateOperatorList leftList = flattenLogicExpression(infixExpression.Left);
            PredicateTemplateOperatorList rightList = flattenLogicExpression(infixExpression.Right);
            if (infixExpression.Operator != ComplexPredicateParser.OperatorType.CONJUNCTION && infixExpression.Operator != ComplexPredicateParser.OperatorType.DISJUNCTION) {
                throw new UnsupportedOperationException("Don't know what to do with binary operator");
            }

            PredicateTemplateOperatorList combined = new PredicateTemplateOperatorList();
            combined.Predicates.addAll(leftList.Predicates);
            combined.Operators.addAll(leftList.Operators);
            combined.Operators.add(infixExpression.Operator);
            combined.Predicates.addAll(rightList.Predicates);
            combined.Operators.addAll(rightList.Operators);
            return combined;

        } else if (ComplexPredicateParser.PrefixUnaryOperatorExpression.class.isInstance(expression)) {

            ComplexPredicateParser.PrefixUnaryOperatorExpression prefixExpression = (ComplexPredicateParser.PrefixUnaryOperatorExpression) expression;
            if (prefixExpression.Operator != ComplexPredicateParser.OperatorType.NEGATION) {
                throw new UnsupportedOperationException("Don't know what to do with unary operator");
            }

            PredicateTemplateOperatorList toNegate = flattenLogicExpression(prefixExpression.Right);
            PredicateTemplateOperatorList negated = new PredicateTemplateOperatorList();
            for (Predicate template : toNegate.Predicates) {
                negated.Predicates.add(new Predicate(template.Name, template.Arguments, !template.IsNegated));
            }

            for (ComplexPredicateParser.OperatorType operator : toNegate.Operators) {

                if (operator == ComplexPredicateParser.OperatorType.DISJUNCTION) {
                    negated.Operators.add(ComplexPredicateParser.OperatorType.CONJUNCTION);
                } else if (operator == ComplexPredicateParser.OperatorType.CONJUNCTION) {
                    negated.Operators.add(ComplexPredicateParser.OperatorType.DISJUNCTION);
                } else {
                    throw new UnsupportedOperationException("Don't know how to negate operator");
                }
            }

            return negated;

        } else if (ComplexPredicateParser.FunctionExpression.class.isInstance(expression)) {

            ComplexPredicateParser.FunctionExpression functionExpression = (ComplexPredicateParser.FunctionExpression) expression;
            String functionName = ((ComplexPredicateParser.NameExpression) functionExpression.FunctionName).Name;
            String[] arguments = new String[functionExpression.Arguments.length];
            for (int i = 0; i < functionExpression.Arguments.length; ++i) {
                arguments[i] = ((ComplexPredicateParser.NameExpression) functionExpression.Arguments[i]).Name;
            }
            Predicate template = new Predicate(functionName, new ArrayList<>(Arrays.asList(arguments)), false);
            PredicateTemplateOperatorList templateList = new PredicateTemplateOperatorList();
            templateList.Predicates.add(template);
            return templateList;

        } else {

            throw new UnsupportedOperationException("Don't know what to do with expression");

        }

    }

}
