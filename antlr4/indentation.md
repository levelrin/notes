## Goal

We want to parse the text in which indentation matters.

For example, we have the following text:
```
a  :
  b:
    c:
      apple  ;
      banana;
      d:
        orange;
  kiwi;
```

And we want to format it to this:
```
a:
  b:
    c:
      apple;
      banana;
      d:
        orange;
  kiwi;
```

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

tokens {DEDENT}

statements
    : statement+
    ;

statement
    : (simpleStatement|subStatement)
    ;

simpleStatement
    : NAME SEMICOLON
    ;

subStatement
    : NAME COLON INDENT statements DEDENT
    ;

NAME: [a-z]+;
COLON: ':';
SEMICOLON: ';';
INDENT: [\r\n]+' '*;
SPACES: ' ' -> skip;
```

## Java

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseVisitor;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonToken;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

final class OurGrammarLexer extends GrammarLexer {

    private final Stack<Integer> indents = new Stack<>();

    private final Queue<Token> tokenQueue = new LinkedList<>();

    public OurGrammarLexer(final CharStream input) {
        super(input);
    }

    @Override
    public Token nextToken() {
        if (!this.tokenQueue.isEmpty()) {
            return this.tokenQueue.remove();
        }
        Token token = super.nextToken();
        int type = token.getType();
        if (type == GrammarLexer.INDENT) {
            final String spaces = token.getText().replaceAll("[\r\n]", "");
            final int currentAmount = spaces.length();
            final int lastAmount;
            if (this.indents.empty()) {
                // This will happen at the initial indentation.
                lastAmount = 0;
            } else {
                lastAmount = this.indents.peek();
            }
            if (currentAmount == lastAmount) {
                final Token nextToken = super.nextToken();
                // It was continuous statements without changes in indentation.
                // For that reason, we just move to the next token with the current indentation.
                token = new CommonToken(nextToken.getType(), token.getText() + nextToken.getText());
            } else if (currentAmount > lastAmount) {
                this.indents.push(currentAmount);
                token = new CommonToken(GrammarParser.INDENT, token.getText());
            } else {
                this.indents.pop();
                token = new CommonToken(GrammarParser.DEDENT, token.getText());
                // It's for the case where multiple dedents occur at the same time.
                while (!this.indents.isEmpty() && currentAmount < this.indents.peek()) {
                    this.indents.pop();
                    // Since we cannot return multiple dedents at once, we put them into the queue.
                    this.tokenQueue.add(new CommonToken(GrammarParser.DEDENT, ""));
                }
            }
        } else if (type == Token.EOF && !this.indents.isEmpty()) {
            // This is a corner case where the file ends without a new line.
            // This may lead to a situation where the number of dedents does not match with the number of indents.
            token = new CommonToken(GrammarParser.DEDENT, "");
        }
        return token;
    }

}

final class GrammarVisitor extends GrammarBaseVisitor<String> {

    @Override
    public String visitStatements(final GrammarParser.StatementsContext context) {
        final List<GrammarParser.StatementContext> statementContexts = context.statement();
        final StringBuilder text = new StringBuilder();
        for (final GrammarParser.StatementContext statement : statementContexts) {
            text.append(this.visit(statement));
        }
        return text.toString();
    }

    @Override
    public String visitStatement(final GrammarParser.StatementContext context) {
        final GrammarParser.SimpleStatementContext simpleStatement = context.simpleStatement();
        final GrammarParser.SubStatementContext subStatement = context.subStatement();
        final StringBuilder text = new StringBuilder();
        if (simpleStatement == null) {
            text.append(this.visit(subStatement));
        } else {
            text.append(this.visit(simpleStatement));
        }
        return text.toString();
    }

    @Override
    public String visitSimpleStatement(final GrammarParser.SimpleStatementContext context) {
        final TerminalNode nameTerminal = context.NAME();
        final TerminalNode semicolon = context.SEMICOLON();
        final StringBuilder text = new StringBuilder();
        text.append(this.visit(nameTerminal))
            .append(this.visit(semicolon));
        return text.toString();
    }

    @Override
    public String visitSubStatement(final GrammarParser.SubStatementContext context) {
        final TerminalNode nameTerminal = context.NAME();
        final TerminalNode colonTerminal = context.COLON();
        final TerminalNode indentTerminal = context.INDENT();
        final GrammarParser.StatementsContext statements = context.statements();
        final TerminalNode dedentTerminal = context.DEDENT();
        final StringBuilder text = new StringBuilder();
        text.append(this.visit(nameTerminal))
            .append(this.visit(colonTerminal))
            .append(this.visit(indentTerminal))
            .append(this.visit(statements))
            .append(this.visit(dedentTerminal));
        return text.toString();
    }

    @Override
    public String visitTerminal(final TerminalNode node) {
        return node.getText();
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = """
            a  :
              b:
                c:
                  apple  ;
                  banana;
                  d:
                    orange;
              kiwi;""";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new OurGrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.statements();
        final GrammarVisitor visitor = new GrammarVisitor();
        final String result = visitor.visit(tree);
        System.out.println(result);
    }

}
```
