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

Actually, the same can be achieved with a listener using an exception.

Then the question is, which one is better?

Here is a speed test:

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseListener;
import com.levelrin.antlr.generated.GrammarBaseVisitor;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.misc.ParseCancellationException;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarVisitor extends GrammarBaseVisitor<Void> {

    @Override
    public Void visitValue(final GrammarParser.ValueContext context) {
        final TerminalNode nameTerminal = context.NAME();
        if (nameTerminal != null) {
            final String nameText = nameTerminal.getText();
            System.out.println("item: " + nameText);
            if ("one".equals(nameText)) {
                throw new ParseCancellationException("Found `one`, stopping traversal.");
            }
        }
        return this.visitChildren(context);
    }

}

final class GrammarListener extends GrammarBaseListener {

    @Override
    public void enterValue(final GrammarParser.ValueContext context) {
        final TerminalNode nameTerminal = context.NAME();
        if (nameTerminal != null) {
            final String nameText = nameTerminal.getText();
            System.out.println("item: " + nameText);
            if ("one".equals(nameText)) {
                throw new ParseCancellationException("Found `one`, stopping traversal.");
            }
        }
    }

}

public class Main {

    public static void main(final String... args) {
        final int testAmount = 1000;
        System.out.println("Test the visitor.");
        final long visitorStartTime = System.nanoTime();
        for (int index = 0; index < testAmount; index++) {
            controlByVisitor();
        }
        final long visitorEndTime = System.nanoTime();
        final long visitorDurationInNanoseconds = visitorEndTime - visitorStartTime;
        final double visitorDurationInMilliseconds = visitorDurationInNanoseconds / 1_000_000.0;
        System.out.println("Test the listener.");
        final long listenerStartTime = System.nanoTime();
        for (int index = 0; index < testAmount; index++) {
            controlByListener();
        }
        final long listenerEndTime = System.nanoTime();
        final long listenerDurationInNanoseconds = listenerEndTime - listenerStartTime;
        final double listenerDurationInMilliseconds = listenerDurationInNanoseconds / 1_000_000.0;
        System.out.println("Test time of the visitor: " + visitorDurationInMilliseconds);
        System.out.println("Test time of the listener: " + listenerDurationInMilliseconds);
    }

    private static void controlByVisitor() {
        final String originalText = "[apple, banana, [one, two, [uno, dos, tres]]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.list();
        final GrammarVisitor visitor = new GrammarVisitor();
        try {
            visitor.visit(tree);
        } catch (final ParseCancellationException ex) {}
    }

    private static void controlByListener() {
        final String originalText = "[apple, banana, [one, two, [uno, dos, tres]]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.list();
        final GrammarListener listener = new GrammarListener();
        try {
            ParseTreeWalker.DEFAULT.walk(listener, tree);
        } catch (final ParseCancellationException ex) {}
    }

}
```

The last two lines in the console output say:
> Test time of the visitor: 117.6497
> 
> Test time of the listener: 34.6675

As we can see, the listener was significantly faster.

According to ChatGPT, the `ParseTreeWalker.DEFAULT.walk` method is highly optimized, which doesn't rely on recursion.

In other words, the visitor was slower because it traverses the tree through recursion.

Therefore, it might be better to use a listener unless we need more precise control over the parse flow without using an exception.
