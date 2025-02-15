## Goal

We want to define a lexer rule based on the negation of the sequence of characters.

For example, we want to match all characters except the string `apple`.

Unfortunately, Antlr4 does not support the negation of a sequence like this: `ANYTHING_EXCEPT_APPLE: ~('apple')+;`.

A workaround is to use a custom lexer based on the generated one.

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

// Although we don't define the rule for the token `ANYTHING_EXCEPT_APPLE`,
// we can just declare the token name like this so that the parser rule can refer to that token.
tokens {ANYTHING_EXCEPT_APPLE}

root
    : component*
    ;

component
    : APPLE
    | ANYTHING_EXCEPT_APPLE
    ;

APPLE: 'apple';
ANY_CHAR: .;
```

## Java

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseListener;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import java.util.ArrayList;
import java.util.List;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonToken;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.antlr.v4.runtime.tree.TerminalNode;

final class OurGrammarLexer extends GrammarLexer {

    private final List<Token> tokenQueue = new ArrayList<>(1);

    public OurGrammarLexer(final CharStream input) {
        super(input);
    }

    @Override
    public Token nextToken() {
        if (!this.tokenQueue.isEmpty()) {
            return this.tokenQueue.remove(0);
        }
        final StringBuilder text = new StringBuilder();
        Token token;
        int type;
        while (true) {
            token = super.nextToken();
            type = token.getType();
            if (type == Token.EOF || type == GrammarLexer.APPLE) {
                break;
            }
            text.append(token.getText());
        }
        if (!text.isEmpty()) {
            this.tokenQueue.add(token);
            // Note that `ANYTHING_EXCEPT_APPLE` is from the parser class, not from the lexer class.
            token = new CommonToken(GrammarParser.ANYTHING_EXCEPT_APPLE, text.toString());
        }
        return token;
    }

}

final class GrammarListener extends GrammarBaseListener {

    @Override
    public void enterComponent(final GrammarParser.ComponentContext context) {
        final TerminalNode appleTerminal = context.APPLE();
        final TerminalNode anythingExceptAppleTerminal = context.ANYTHING_EXCEPT_APPLE();
        if (appleTerminal == null) {
            System.out.println("Anything: " + anythingExceptAppleTerminal.getText());
        } else {
            System.out.println("Apple: " + appleTerminal.getText());
        }
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = "aaaappleaaa";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new OurGrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.root();
        final GrammarListener listener = new GrammarListener();
        ParseTreeWalker.DEFAULT.walk(listener, tree);
    }

}

```
