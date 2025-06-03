## Goal

From a visitor method, we may want to check the structure of the parent node.

In other words, we may want to climb up the parse tree.

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

list
    : OPEN_BRACKET items? CLOSE_BRACKET
    ;

items
    : item (COMMA item)*
    ;

item
    : NAME
    | list
    ;

OPEN_BRACKET: '[';
CLOSE_BRACKET: ']';
COMMA: ',';
NAME: [a-z]+;
WS: [ \t\r\n]+ -> skip;
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
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarVisitor extends GrammarBaseVisitor<String> {

    @Override
    public String visitList(final GrammarParser.ListContext context) {
        System.out.println("Current list: " + context.getText());
        ParserRuleContext parent = context.getParent();
        while (parent != null) {
            System.out.println("Current parent type: " + parent.getClass().getSimpleName());
            if (parent instanceof GrammarParser.ListContext) {
                break;
            }
            parent = parent.getParent();
        }
        if (parent != null) {
            System.out.println("Parent List: " + parent.getText());
        }
        System.out.println();

        final TerminalNode openBracketTerminal = context.OPEN_BRACKET();
        final GrammarParser.ItemsContext itemsContext = context.items();
        final TerminalNode closeBracketTerminal = context.CLOSE_BRACKET();
        final StringBuilder text = new StringBuilder();
        text.append(this.visit(openBracketTerminal));
        if (itemsContext != null) {
            text.append(this.visit(itemsContext));
        }
        text.append(this.visit(closeBracketTerminal));
        return text.toString();
    }

    @Override
    public String visitItems(final GrammarParser.ItemsContext context) {
        final List<GrammarParser.ItemContext> itemContexts = context.item();
        final List<TerminalNode> commaTerminals = context.COMMA();
        final StringBuilder text = new StringBuilder();
        final GrammarParser.ItemContext firstItemContext = itemContexts.get(0);
        text.append(this.visit(firstItemContext));
        for (int index = 0; index < commaTerminals.size(); index++) {
            final TerminalNode commaTerminal = commaTerminals.get(index);
            final GrammarParser.ItemContext itemContext = itemContexts.get(index + 1);
            text.append(this.visit(commaTerminal))
                .append(' ')
                .append(this.visit(itemContext));
        }
        return text.toString();
    }

    @Override
    public String visitItem(final GrammarParser.ItemContext context) {
        final TerminalNode nameTerminal = context.NAME();
        final GrammarParser.ListContext listContext = context.list();
        final StringBuilder text = new StringBuilder();
        if (nameTerminal != null) {
            text.append(this.visit(nameTerminal));
        } else if (listContext != null) {
            text.append(this.visit(listContext));
        }
        return text.toString();
    }

    @Override
    public String visitTerminal(final TerminalNode node) {
        return node.getText();
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = "[apple, banana, [one, two]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.list();
        final GrammarVisitor visitor = new GrammarVisitor();
        final String result = visitor.visit(tree);
        System.out.println(result);
    }

}
```

Here is the console output:
```
Current list: [apple,banana,[one,two]]

Current list: [one,two]
Current parent type: ItemContext
Current parent type: ItemsContext
Current parent type: ListContext
Parent List: [apple,banana,[one,two]]

[apple, banana, [one, two]]
```
