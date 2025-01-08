## Goal

Here is the input:
```
// one
uno = 1;

// two
dos = 2;
```

We want to change it to the following:
```
// one
uno -> 1

// two
dos -> 2
```

The challenge is that whitespaces and comment lines are skipped in the grammar.

However, we want to preserve skipped tokens.

Here is the grammar:
```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

variableDeclarations
    : variableDeclaration+
    ;

variableDeclaration
    : NAME '=' NUMBER ';'
    ;

NAME: [a-z]+;
NUMBER: [0-9]+;
COMMENT: '//' ~[\r\n]* -> skip;
WHITESPACE: [ \t\r\n] -> skip;
```

## Grammar Modification

We can put whitespaces and comments into hidden channels instead of just skipping them like this:
```g4
COMMENT: '//' ~[\r\n]* -> channel(1);
WHITESPACE: [ \t\r\n] -> channel(2);
```

## Java

And here is how we can use hidden channels in the parser:
```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseVisitor;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.RuleNode;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarVisitor extends GrammarBaseVisitor<String> {

    private final CommonTokenStream tokens;

    public GrammarVisitor(final CommonTokenStream tokens) {
        this.tokens = tokens;
    }

    @Override
    public String visitVariableDeclarations(final GrammarParser.VariableDeclarationsContext context) {
        final List<GrammarParser.VariableDeclarationContext> variableDeclarationContexts = context.variableDeclaration();
        final StringBuilder result = new StringBuilder();
        for (final GrammarParser.VariableDeclarationContext variableDeclarationContext : variableDeclarationContexts) {
            result.append(this.visit(variableDeclarationContext));
        }
        return result.toString();
    }

    @Override
    public String visitVariableDeclaration(final GrammarParser.VariableDeclarationContext context) {
        final TerminalNode name = context.NAME();
        final TerminalNode number = context.NUMBER();
        final StringBuilder result = new StringBuilder();
        result.append(this.visit(name));
        result.append(" ->");
        result.append(this.visit(number));
        return result.toString();
    }

    @Override
    public String visitChildren(final RuleNode node) {
        throw new UnsupportedOperationException(
            String.format(
                "The following rule is not implemented yet: %s text: %s",
                node.getClass(),
                node.getText()
            )
        );
    }

    @Override
    public String visit(final ParseTree tree) {
        return tree.accept(this);
    }

    @Override
    public String visitTerminal(final TerminalNode node) {
        final String text = node.getText();
        final int tokenIndex = node.getSymbol().getTokenIndex();
        final int commentChannel = 1;
        final int whitespaceChannel = 2;
        // The hidden token list can be null.
        final List<Token> comments = this.tokens.getHiddenTokensToLeft(tokenIndex, commentChannel);
        final List<Token> whitespaces = this.tokens.getHiddenTokensToLeft(tokenIndex, whitespaceChannel);
        final Map<Integer, String> hiddenTokenMap = new TreeMap<>();
        final StringBuilder result = new StringBuilder();
        if (comments != null) {
            for (final Token comment : comments) {
                hiddenTokenMap.put(
                    comment.getStartIndex(),
                    comment.getText()
                );
            }
        }
        if (whitespaces != null) {
            for (final Token whitespace : whitespaces) {
                hiddenTokenMap.put(
                    whitespace.getStartIndex(),
                    whitespace.getText()
                );
            }
        }
        for (final String hiddenTokenText : hiddenTokenMap.values()) {
            result.append(hiddenTokenText);
        }
        result.append(text);
        return result.toString();
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = """
            // one
            uno = 1;
            
            // two
            dos = 2;
            """;
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.variableDeclarations();
        final GrammarVisitor visitor = new GrammarVisitor(tokens);
        final String result = visitor.visit(tree);
        System.out.printf("result:%n%s%n", result);
    }

}
```
