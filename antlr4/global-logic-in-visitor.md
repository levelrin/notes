## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

numbers
    : number (SPACE number)*
    ;

number
    : one
    | two
    | three
    ;

one
    : ONE
    ;

two
    : TWO
    ;

three
    : THREE
    ;

ONE: 'one';
TWO: 'two';
THREE: 'three';
SPACE: [ ];
```

## Java

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseVisitor;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import java.util.List;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.RuleNode;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarVisitor extends GrammarBaseVisitor<String> {

    @Override
    public String visitNumbers(final GrammarParser.NumbersContext context) {
        final StringBuilder result = new StringBuilder();
        final List<GrammarParser.NumberContext> numberContexts = context.number();
        final GrammarParser.NumberContext firstNumberContext = numberContexts.get(0);
        result.append(this.visit(firstNumberContext));
        for (int index = 1; index < numberContexts.size(); index++) {
            result.append(" ");
            result.append(this.visit(numberContexts.get(index)));
        }
        return result.toString();
    }

    @Override
    public String visitNumber(final GrammarParser.NumberContext context) {
        final GrammarParser.OneContext oneContext = context.one();
        final GrammarParser.TwoContext twoContext = context.two();
        final GrammarParser.ThreeContext threeContext = context.three();
        final StringBuilder result = new StringBuilder();
        if (oneContext != null) {
            result.append(this.visit(oneContext));
        }
        if (twoContext != null) {
            result.append(this.visit(twoContext));
        }
        if (threeContext != null) {
            result.append(this.visit(threeContext));
        }
        return result.toString();
    }

    @Override
    public String visitOne(final GrammarParser.OneContext context) {
        return this.visit(context.ONE());
    }

    @Override
    public String visitTwo(final GrammarParser.TwoContext context) {
        return this.visit(context.TWO());
    }

    @Override
    public String visitThree(final GrammarParser.ThreeContext context) {
        return this.visit(context.THREE());
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
        System.out.printf("visit rule: %s%n", tree.getClass().getSimpleName());
        return tree.accept(this);
    }

    @Override
    public String visitTerminal(final TerminalNode node) {
        System.out.printf("visit terminal: %s%n", node.getText());
        return node.getText();
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = "one two three";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.numbers();
        final GrammarVisitor visitor = new GrammarVisitor();
        final String result = visitor.visit(tree);
        System.out.printf("result: %s%n", result);
    }

}
```

Output:
```
visit rule: NumbersContext
visit rule: NumberContext
visit rule: OneContext
visit rule: TerminalNodeImpl
visit terminal: one
visit rule: NumberContext
visit rule: TwoContext
visit rule: TerminalNodeImpl
visit terminal: two
visit rule: NumberContext
visit rule: ThreeContext
visit rule: TerminalNodeImpl
visit terminal: three
result: one two three
```
