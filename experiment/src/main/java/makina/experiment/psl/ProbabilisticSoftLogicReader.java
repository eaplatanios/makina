package makina.experiment.psl;

import com.google.common.collect.ImmutableList;
import makina.experiment.psl.parser.ComplexPredicateParser;
import makina.learn.logic.formula.*;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import makina.experiment.psl.parser.PrattParserExpression;
import makina.learn.logic.LogicManager;
import makina.learn.logic.ProbabilisticSoftLogic;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.zip.DataFormatException;

/**
 * Created by Dan Schwartz 4/27/15
 * Reads files associated with probabilistic soft logic
 */
public class ProbabilisticSoftLogicReader {

    private ProbabilisticSoftLogicReader() {}

    public static void readGroundingsAndAddToManager(
            LogicManager logicManager,
            EntityType entityType,
            String predicateName,
            boolean isIgnoreValues,
            String filename) throws DataFormatException, IOException {

        BufferedReader br = null;

        try {

            File file = new File(filename);

            br = new BufferedReader(new FileReader(file));
            readGroundingsAndAddToManager(logicManager, predicateName, isIgnoreValues, br);

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

    public static void readGroundingsAndAddToManager(
            LogicManager logicManager,
            String predicateName,
            boolean isIgnoreValues,
            BufferedReader reader) throws DataFormatException, IOException {

        String line;

        int lineNumber = 0;
        boolean hasValues = false;
        while ((line = reader.readLine()) != null) {

            String[] lineFields = line.split("\t");
            if (lineNumber == 0 && lineFields.length == 3) {
                hasValues = true;
            }

            double value = 1;
            if (hasValues) {
                if (lineFields.length != 3) {
                    throw new DataFormatException("If any line has a computeValue, all lines must have a computeValue.  Bad format on line: " + (lineNumber + 1));
                }
                value = Double.parseDouble(lineFields[2]);
            } else if (lineFields.length != 2) {
                throw new DataFormatException("Bad format on line: " + (lineNumber + 1));
            }

            ImmutableList.Builder<String> currentGrounding = ImmutableList.builder();
            for (int indexEntity = 0; indexEntity < 2; ++indexEntity) {
                String entity = lineFields[indexEntity];
                entity = entity.trim();
                if (entity.isEmpty()) {
                    throw new DataFormatException("Empty entity on line: " + (lineNumber + 1));
                }
                currentGrounding.add(entity);
            }

            Predicate predicate = logicManager.getPredicate(predicateName);
            if (isIgnoreValues) {
                logicManager.addGroundPredicate(
                        predicate,
                        currentGrounding.build().stream().map(Long::parseLong).collect(Collectors.toList()),
                        logicManager.logic().trueValue()
                );
            } else {
                logicManager.addGroundPredicate(
                        predicate,
                        currentGrounding.build().stream().map(Long::parseLong).collect(Collectors.toList()),
                        value
                );
            }

            ++lineNumber;
        }
    }

    public static List<ProbabilisticSoftLogic.LogicRule> readRules(BufferedReader reader,
                                                                   LogicManager logicManager,
                                                                   List<Variable> variables) throws DataFormatException, IOException {

        List<ProbabilisticSoftLogic.LogicRule> result = new ArrayList<>();

        String line;

        int lineNumber = 0;
        while ((line = reader.readLine()) != null) {

            line = line.trim();
            if (line.isEmpty()) {
                ++lineNumber;
                continue;
            }

            // DBC: Avoid having to remove the first line of the model file every time
            if (line.equals("Model:")) {
                ++lineNumber;
                continue;
            }

            if (!line.startsWith("{")) {
                throw new DataFormatException("Expected rules of the form {weight} body >> head : no leading curly brace found at line " + (lineNumber + 1));
            }

            int indexEndCurly = line.indexOf("}");
            if (indexEndCurly < 0) {
                throw new DataFormatException("Expected rules of the form {weight} body >> head : no closing curly brace found at line " + (lineNumber + 1));
            }

            String weightOrConstraint = line.substring(1, indexEndCurly);

            // DBC: Deal with the "W=" included in the rule weight in the pre-grounded rules
            int indexW = weightOrConstraint.indexOf("W=");
            if (indexW >= 0) {
                weightOrConstraint = weightOrConstraint.substring(2);
            }

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

            // DBC: modified to deal with pre-groudned rules, which only have a head (and no '>>')
            PrattParserExpression headExpression = null;
            PrattParserExpression bodyExpression = null;

            if (headBodyArr.length == 1) {
                headExpression = ComplexPredicateParser.parseRule(headBodyArr[0]);
            }
            else if (headBodyArr.length == 2) {
                headExpression = ComplexPredicateParser.parseRule(headBodyArr[1]);
                bodyExpression = ComplexPredicateParser.parseRule(headBodyArr[0]);
            }
            else {
                throw new DataFormatException("Expected rules of the form {weight} body >> head (or body only): number of head/body parts is wrong at line " + (lineNumber + 1));
                //throw new DataFormatException("Expected rules of the form {weight} body >> head : number of head/body parts is wrong at line " + (lineNumber + 1));
            }

            PredicateTemplateOperatorList flattenedHead = flattenLogicExpression(headExpression);
            for (ComplexPredicateParser.OperatorType operator : flattenedHead.Operators) {
                if (operator != ComplexPredicateParser.OperatorType.DISJUNCTION) {
                    throw new UnsupportedOperationException("Only disjunctions are allowed in the head of a rule");
                }
            }

            PredicateTemplateOperatorList flattenedBody = null;
            if (headBodyArr.length > 1) {
                flattenedBody = flattenLogicExpression(bodyExpression);
                for (ComplexPredicateParser.OperatorType operator : flattenedBody.Operators) {
                    if (operator != ComplexPredicateParser.OperatorType.CONJUNCTION) {
                        throw new UnsupportedOperationException("Only conjunctions are allowed in the body of a rule");
                    }
                }
            }

            if (headBodyArr.length == 1) {
                // DBC: Create a rule with only a head (messes up the name, but that's okay)
                throw new UnsupportedOperationException();
//                ProbabilisticSoftLogicProblem.Rule rule = new ProbabilisticSoftLogicProblem.Rule(weight, power, ImmutableList.copyOf(flattenedHead.Predicates));
//                result.add(rule);
            }
            else if (headBodyArr.length == 2) {
                List<TemporaryPredicate> body = ImmutableList.copyOf(flattenedBody.Predicates);
                List<TemporaryPredicate> head = ImmutableList.copyOf(flattenedHead.Predicates);

                List<Formula> bodyParts = new ArrayList<>();
                for (int i = 0; i < body.size(); ++i) {
                    Predicate predicate = logicManager.getPredicate(body.get(i).Name);
                    List<Variable> predicateArguments = body.get(i).Arguments
                            .stream()
                            .map(argument -> variables.stream().filter(variable ->  variable.getName().equals(argument)).findFirst().get())
                            .collect(Collectors.toList());
                    if (body.get(i).IsNegated)
                        bodyParts.add(new Negation(new Atom(predicate, predicateArguments)));
                    else
                        bodyParts.add(new Atom(predicate, predicateArguments));
                }
                List<Formula> headParts = new ArrayList<>();
                for (int i = 0; i < head.size(); ++i) {
                    Predicate predicate = logicManager.getPredicate(head.get(i).Name);
                    List<Variable> predicateArguments = head.get(i).Arguments.stream()
                            .map(argument -> variables.stream().filter(variable -> variable.getName().equals(argument)).findFirst().get())
                            .collect(Collectors.toList());
                    if (head.get(i).IsNegated)
                        headParts.add(new Negation(new Atom(predicate, predicateArguments)));
                    else
                        headParts.add(new Atom(predicate, predicateArguments));
                }
                ProbabilisticSoftLogic.LogicRule logicRule = new ProbabilisticSoftLogic.LogicRule(bodyParts, headParts, power, weight);
                result.add(logicRule);
            }

            ++lineNumber;

        }

        return result;

    }

    private static class PredicateTemplateOperatorList {

        public final ArrayList<ComplexPredicateParser.OperatorType> Operators = new ArrayList<>();
        public final ArrayList<TemporaryPredicate> Predicates = new ArrayList<>();

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
            for (TemporaryPredicate template : toNegate.Predicates) {
                negated.Predicates.add(new TemporaryPredicate(template.Name, template.Arguments, !template.IsNegated));
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
            ImmutableList.Builder<String> arguments = ImmutableList.builder();
            for (int i = 0; i < functionExpression.Arguments.length; ++i) {
                arguments.add(((ComplexPredicateParser.NameExpression) functionExpression.Arguments[i]).Name);
            }
            TemporaryPredicate template = new TemporaryPredicate(functionName, arguments.build(), false);
            PredicateTemplateOperatorList templateList = new PredicateTemplateOperatorList();
            templateList.Predicates.add(template);
            return templateList;

        } else {

            throw new UnsupportedOperationException("Don't know what to do with expression");

        }

    }

    public static class TemporaryPredicate implements Serializable {

        public TemporaryPredicate(String name, ImmutableList<String> arguments, boolean isNegated) {
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

            TemporaryPredicate rhs = (TemporaryPredicate)other;

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

}
