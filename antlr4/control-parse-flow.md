## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

list
    : OPEN_BRACKET value (COMMA value)* CLOSE_BRACKET
    ;

value
    : NAME
    | list
    ;

COMMA: ',';
OPEN_BRACKET: '[';
CLOSE_BRACKET: ']';
NAME: [a-z] ([a-z0-9-]* [a-z0-9])?;
WS: [ \t\r\n]+ -> skip;
```

## Java

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseVisitor;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.misc.ParseCancellationException;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarVisitor extends GrammarBaseVisitor<Void> {

    @Override
    public Void visitValue(final GrammarParser.ValueContext context) {
        final TerminalNode nameTerminal = context.NAME();
        if (nameTerminal != null) {
            final String nameText = nameTerminal.getText();
            System.out.println("item: " + nameText);
            if ("one".equals(nameText)) {
                // Returning null only stops the current parsing branch.
                // In this case, it won't stop the entire traversal.
                // return null;

                // Although it's not a good coding practice, throwing an exception is the most reliable way to stop parsing.
                throw new ParseCancellationException("Found `one`, stopping traversal.");
            }
        }
        return this.visitChildren(context);
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = "[apple, banana, [one, two, [uno, dos, tres]]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.list();
        final GrammarVisitor visitor = new GrammarVisitor();
        try {
            visitor.visit(tree);
        } catch (final ParseCancellationException ex) {
            // Do nothing.
        }
    }

}
```

This was the console output:
```
item: apple
item: banana
item: one
```
