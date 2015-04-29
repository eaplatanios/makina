package org.platanios.experiment.psl.parser;

import java.util.*;
import java.util.zip.DataFormatException;

/*
 * Created by Dan Schwartz on 4/25/2015.
 * Primarily based on Pratt parser implementation described at
 * http://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/
 */
public class PrattParser {

    protected static class TokenIDTable {

        public TokenIDTable(char ... charactersToTokenize) {
            int nextId = NameToken + 1;
            this.charToToken = new HashMap<>();
            this.tokenToChar = new HashMap<>();
            for (char character : charactersToTokenize) {
                if (!charToToken.containsKey(character)) {
                    charToToken.put(character, nextId);
                    tokenToChar.put(nextId, character);
                    ++nextId;
                }
            }
        }

        public int GetTokenId(char character) {
            return this.charToToken.getOrDefault(character, EndToken);
        }
        public char GetCharFromTokenId(int tokenId) { return this.tokenToChar.getOrDefault(tokenId, '\0'); }

        private final HashMap<Character, Integer> charToToken;
        private final HashMap<Integer, Character> tokenToChar;
        public static final int EndToken = 0;
        public static final int NameToken = 1;
    }

    protected interface PrefixParselet {
        PrattParserExpression parse(ParseState state, Token token) throws DataFormatException;
    }

    protected interface InfixParselet {
        PrattParserExpression parse(ParseState state, PrattParserExpression left, Token token) throws DataFormatException;
        int getPrecedence();
    }

    public PrattParser(
            TokenIDTable tokenTable,
            List<AbstractMap.SimpleEntry<Integer, PrefixParselet>> prefixParselets,
            List<AbstractMap.SimpleEntry<Integer, InfixParselet>> infixParselets) {

        this.TokenTable = tokenTable;
        this.PrefixParselets = new HashMap<>();
        this.InfixParselets = new HashMap<>();

        for (AbstractMap.SimpleEntry<Integer, PrefixParselet> prefixParselet : prefixParselets) {
            if (this.PrefixParselets.containsKey(prefixParselet.getKey())) {
                throw new UnsupportedOperationException("Multiple prefix parselets with the same key");
            }
            this.PrefixParselets.put(prefixParselet.getKey(), prefixParselet.getValue());
        }

        for (AbstractMap.SimpleEntry<Integer, InfixParselet> infixParselet : infixParselets) {
            if (this.InfixParselets.containsKey(infixParselet.getKey())) {
                throw new UnsupportedOperationException("Multiple infix parselets with the same key");
            }
            this.InfixParselets.put(infixParselet.getKey(), infixParselet.getValue());
        }

    }

    public PrattParserExpression parse(String expression) throws DataFormatException {
        Lexer lexer = new Lexer(expression);
        ParseState state = new ParseState(lexer);
        return state.parseExpression();
    }

    protected static class Token {

        public Token(int tokenID, String text) {
            this.TokenID = tokenID;
            this.Text = text;
        }

        public final int TokenID;
        public final String Text;

    }

    private class Lexer implements Iterator<Token> {

        public Lexer(String expression) {
            this.expression = expression;
            this.index = 0;
        }

        @Override
        public boolean hasNext() {
            // the parse does lookahead, so we return dummy tokens if necessary
            return true;
        }

        @Override
        public Token next() {

            while (this.index < this.expression.length()) {

                char current = this.expression.charAt(this.index++);
                if (Character.isWhitespace(current)) {
                    continue;
                }

                int simpleType = PrattParser.this.TokenTable.GetTokenId(current);
                if (simpleType != TokenIDTable.EndToken) {
                    return new Token(simpleType, Character.toString(current));
                } else {
                    int startIndex = this.index - 1;
                    char nameChar = this.expression.charAt(this.index++);
                    while (this.index < this.expression.length() &&
                            !Character.isWhitespace(nameChar) &&
                            PrattParser.this.TokenTable.GetTokenId(nameChar) == TokenIDTable.EndToken) {
                        nameChar = this.expression.charAt(this.index++);
                    }
                    --this.index;
                    return new Token(TokenIDTable.NameToken, this.expression.substring(startIndex, this.index));
                }
            }

            return new Token(TokenIDTable.EndToken, null);

        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

        private int index;
        private final String expression;

    }

    public class ParseState {

        public ParseState(Iterator<Token> tokens) {
            this.IteratorTokens = tokens;
            this.ReadTokens = new ArrayList<>();
        }

        public PrattParserExpression parseExpression(int precedence) throws DataFormatException {
            Token token = this.consume();
            PrefixParselet prefix = PrattParser.this.PrefixParselets.get(token.TokenID);

            if (prefix == null) {
                throw new DataFormatException("Could not parse:" + token.Text);
            }

            PrattParserExpression left = prefix.parse(this, token);

            while (precedence < this.getPrecedence()) {
                token = this.consume();
                InfixParselet infix = PrattParser.this.InfixParselets.get(token.TokenID);
                left = infix.parse(this, left, token);
            }

            return left;
        }

        public PrattParserExpression parseExpression() throws DataFormatException {
            return this.parseExpression(0);
        }

        public boolean consumeOnMatch(int expectedTokenId) {

            Token token = this.lookAhead(0);
            if (token.TokenID != expectedTokenId) {
                return false;
            }

            this.consume();
            return true;

        }

        public Token consume(int expectedTokenId) throws DataFormatException {

            Token token = this.lookAhead(0);
            if (token.TokenID != expectedTokenId) {
                throw new DataFormatException("Expected token " + Character.toString(TokenTable.GetCharFromTokenId(expectedTokenId)) + " and got " + token.Text);
            }

            return this.consume();

        }

        public Token consume() {
            this.lookAhead(0);
            return this.ReadTokens.remove(0);
        }

        public Token lookAhead(int distance) {
            while (distance >= this.ReadTokens.size()) {
                this.ReadTokens.add(this.IteratorTokens.next());
            }

            return this.ReadTokens.get(distance);
        }

        public int getPrecedence() {

            InfixParselet tempParselet = PrattParser.this.InfixParselets.get(this.lookAhead(0).TokenID);

            if (tempParselet != null) {
                return tempParselet.getPrecedence();
            }

            return 0;
        }

        private final Iterator<Token> IteratorTokens;
        private final List<Token> ReadTokens;

    }

    private final TokenIDTable TokenTable;
    private final Map<Integer, PrefixParselet> PrefixParselets;
    private final Map<Integer, InfixParselet> InfixParselets;

}
