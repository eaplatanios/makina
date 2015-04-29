package org.platanios.experiment.psl.parser;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.DataFormatException;

/**
 * Parser that is specific to logic rules, uses PrattParser to parse
 */
public class ComplexPredicateParser {

    public static class InfixBinaryOperatorExpression implements PrattParserExpression {

        public InfixBinaryOperatorExpression(PrattParserExpression left, OperatorType operator, PrattParserExpression right) {

            this.Left = left;
            this.Operator = operator;
            this.Right = right;

        }

        @Override
        public String toString() {

            return "(" + this.Left.toString() + " " + this.Operator.toString() + " " + this.Right.toString() + ")";

        }

        public final PrattParserExpression Left;
        public final OperatorType Operator;
        public final PrattParserExpression Right;

    }

    public static class PrefixUnaryOperatorExpression implements PrattParserExpression {

        public PrefixUnaryOperatorExpression(OperatorType operator, PrattParserExpression right) {
            this.Operator = operator;
            this.Right = right;
        }

        @Override
        public String toString() {

            return this.Operator.toString() + this.Right.toString();

        }

        public final OperatorType Operator;
        public final PrattParserExpression Right;

    }

    public static class FunctionExpression implements PrattParserExpression {

        public FunctionExpression(PrattParserExpression functionName, PrattParserExpression[] arguments) {
            this.FunctionName = functionName;
            this.Arguments = arguments;
        }

        @Override
        public String toString() {

            StringBuilder sb = new StringBuilder();
            sb.append(this.FunctionName.toString());
            sb.append("(");
            for (int i = 0; i < this.Arguments.length; ++i) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(this.Arguments[i].toString());
            }
            sb.append(")");

            return sb.toString();

        }

        public final PrattParserExpression FunctionName;
        public final PrattParserExpression[] Arguments;

    }

    public static class NameExpression implements PrattParserExpression {

        public NameExpression(String name) {

            this.Name = name;

        }

        @Override
        public String toString() {
            return this.Name;
        }

        public final String Name;

    }

    public enum OperatorType {
        DISJUNCTION,
        CONJUNCTION,
        NEGATION;

        @Override
        public String toString() {
            switch (this) {
                case DISJUNCTION:
                    return "|";
                case CONJUNCTION:
                    return "&";
                case NEGATION:
                    return "~";
                default:
                    throw new UnsupportedOperationException();
            }
        }
    }

    public static PrattParserExpression parseRule(String rule) throws DataFormatException {

        return PlattLogicRuleParser.parse(rule);

    }

    private static class PrecedenceTable {
        public static final int DISJUNCTION = 1;
        public static final int CONJUNCTION = 2;
        public static final int NEGATION = 3;
        public static final int FUNCTION = 4;
    }

    private static class PrefixOperatorParselet implements PrattParser.PrefixParselet {

        public PrefixOperatorParselet(int precedence) {
            this.Precedence = precedence;
        }

        public PrattParserExpression parse(PrattParser.ParseState state, PrattParser.Token token) throws DataFormatException {
            PrattParserExpression operand = state.parseExpression(this.Precedence);
            if (token.TokenID == TokenIDTilde) {
                return new PrefixUnaryOperatorExpression(OperatorType.NEGATION, operand);
            }
            else {
                throw new DataFormatException("No known operator associated with:" + token.Text);
            }
        }

        public final int Precedence;

    }

    private static class InfixOperatorParselet implements PrattParser.InfixParselet {

        public InfixOperatorParselet(int precedence, boolean isRightAssociative) {
            this.Precedence = precedence;
            this.IsRightAssociative = isRightAssociative;
        }

        public PrattParserExpression parse(PrattParser.ParseState state, PrattParserExpression left, PrattParser.Token token) throws DataFormatException {
            OperatorType opType;
            if (token.TokenID == TokenIDAmpersand) {
                opType = OperatorType.CONJUNCTION;
            }
            else if (token.TokenID == TokenIDPipe) {
                opType = OperatorType.DISJUNCTION;
            } else {
                throw new DataFormatException("No operator associated with token:" + token.Text);
            }
            PrattParserExpression right = state.parseExpression();

            return new InfixBinaryOperatorExpression(left, opType, right);
        }

        public int getPrecedence() {
            return this.Precedence;
        }

        public final int Precedence;
        public final boolean IsRightAssociative;

    }

    private static class GroupParselet implements PrattParser.PrefixParselet {

        public PrattParserExpression parse(PrattParser.ParseState state, PrattParser.Token token) throws DataFormatException {

            PrattParserExpression innerExpression = state.parseExpression();
            state.consume(TokenIDCloseParen);
            return innerExpression;

        }

    }

    private static class PredicateParselet implements PrattParser.InfixParselet {
        public PrattParserExpression parse(PrattParser.ParseState state, PrattParserExpression left, PrattParser.Token token) throws DataFormatException {
            List<PrattParserExpression> args = new ArrayList<>();
            if (!state.consumeOnMatch(TokenIDCloseParen)) {
                do {
                    // add each argument
                    args.add(state.parseExpression());
                } while (state.consumeOnMatch(TokenIDComma));
                state.consume(TokenIDCloseParen);
            }

            PrattParserExpression[] argsArr = new PrattParserExpression[args.size()];
            return new FunctionExpression(left, args.toArray(argsArr));
        }

        public int getPrecedence() { return PrecedenceTable.FUNCTION; }
    }

    private static class NameParselet implements PrattParser.PrefixParselet {
        public PrattParserExpression parse(PrattParser.ParseState state, PrattParser.Token token) {
            return new NameExpression(token.Text);
        }
    }

    private static final PrattParser.TokenIDTable TokenTable =
            new PrattParser.TokenIDTable('|', '&', '~', '(', ')', ',', '{', '}', '>');
    private static final int TokenIDPipe = TokenTable.GetTokenId('|');
    private static final int TokenIDAmpersand = TokenTable.GetTokenId('&');
    private static final int TokenIDTilde = TokenTable.GetTokenId('~');
    private static final int TokenIDOpenParen = TokenTable.GetTokenId('(');
    private static final int TokenIDCloseParen = TokenTable.GetTokenId(')');
    private static final int TokenIDComma = TokenTable.GetTokenId(',');
    private static final PrattParser PlattLogicRuleParser =
            new PrattParser(
                    TokenTable,
                    new ArrayList<>(Arrays.asList(
                        new AbstractMap.SimpleEntry<>(TokenIDTilde, new PrefixOperatorParselet(PrecedenceTable.NEGATION)),
                        new AbstractMap.SimpleEntry<>(TokenIDOpenParen, new GroupParselet()),
                        new AbstractMap.SimpleEntry<>(PrattParser.TokenIDTable.NameToken, new NameParselet()))),
                    new ArrayList<>(Arrays.asList(
                        new AbstractMap.SimpleEntry<>(TokenIDPipe, new InfixOperatorParselet(PrecedenceTable.DISJUNCTION, false)),
                        new AbstractMap.SimpleEntry<>(TokenIDAmpersand, new InfixOperatorParselet(PrecedenceTable.CONJUNCTION, false)),
                        new AbstractMap.SimpleEntry<>(TokenIDOpenParen, new PredicateParselet())))
                    );
}
