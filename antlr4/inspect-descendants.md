## Goal

Let's say we have the following text:
```
[one, two, three] [apple, banana, [one, two]]
```

We want to format that text like this:
```
[one, two, three]
[
  apple,
  banana,
  [one, two]
]
```

To do that, we need to know if the list is nested before visiting further into the parse tree.

In other words, we need to inspect descendants if there is a list in the parse tree.

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

file
    : statement+
    ;

statement
    : list
    ;

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
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarVisitor extends GrammarBaseVisitor<String> {

    /**
     * Number of spaces for an indentation.
     */
    private static final String INDENT_UNIT = "  ";

    /**
     * As is.
     */
    private int currentIndentLevel;

    @Override
    public String visitFile(final GrammarParser.FileContext context) {
        final List<GrammarParser.StatementContext> statementContexts = context.statement();
        final StringBuilder text = new StringBuilder();
        for (int index = 0; index < statementContexts.size(); index++) {
            final  GrammarParser.StatementContext statementContext = statementContexts.get(index);
            text.append(this.visit(statementContext));
            if (index < statementContexts.size() - 1) {
                this.appendNewLinesAndIndent(text, 1);
            }
        }
        return text.toString();
    }

    @Override
    public String visitList(final GrammarParser.ListContext context) {
        final TerminalNode openBracketTerminal = context.OPEN_BRACKET();
        final GrammarParser.ItemsContext itemsContext = context.items();
        final TerminalNode closeBracketTerminal = context.CLOSE_BRACKET();
        final StringBuilder text = new StringBuilder();
        boolean shouldIndent = false;
        if (itemsContext != null) {
            shouldIndent = this.hasDescendant(itemsContext, GrammarParser.ListContext.class);
        }
        if (shouldIndent) {
            text.append(this.visit(openBracketTerminal));
            this.currentIndentLevel++;
            this.appendNewLinesAndIndent(text, 1);
            text.append(this.visit(itemsContext));
            this.currentIndentLevel--;
            this.appendNewLinesAndIndent(text, 1);
            text.append(this.visit(closeBracketTerminal));
        } else {
            text.append(this.visit(openBracketTerminal));
            if (itemsContext != null) {
                text.append(this.visit(itemsContext));
            }
            text.append(this.visit(closeBracketTerminal));
        }
        return text.toString();
    }

    @Override
    public String visitItems(final GrammarParser.ItemsContext context) {
        final List<GrammarParser.ItemContext> itemContexts = context.item();
        final List<TerminalNode> commaTerminals = context.COMMA();
        final StringBuilder text = new StringBuilder();
        boolean shouldPlaceVertically = this.hasDescendant(context, GrammarParser.ListContext.class);
        final GrammarParser.ItemContext firstItemContext = itemContexts.get(0);
        text.append(this.visit(firstItemContext));
        for (int index = 0; index < commaTerminals.size(); index++) {
            final TerminalNode commaTerminal = commaTerminals.get(index);
            final GrammarParser.ItemContext itemContext = itemContexts.get(index + 1);
            text.append(this.visit(commaTerminal));
            if (shouldPlaceVertically) {
                this.appendNewLinesAndIndent(text, 1);
            } else {
                text.append(' ');
            }
            text.append(this.visit(itemContext));
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

    /**
     * Inspect the descendants to see if the specified type is found in the parse tree.
     * @param subject The search begins from this.
     * @param descendantType The type we are looking for.
     * @return True if a descendant with the specified type is found.
     */
    private boolean hasDescendant(final ParserRuleContext subject, final Class<? extends ParserRuleContext> descendantType) {
        boolean result = false;
        // Note that the `children` attribute only gives us direct children.
        // In other words, it doesn't recursively include all descendants.
        final Queue<ParseTree> queue = new LinkedList<>(subject.children);
        while (!queue.isEmpty()) {
            final ParseTree child = queue.remove();
            if (descendantType.isInstance(child)) {
                result = true;
                break;
            }
            for (int index = 0; index < child.getChildCount(); index++) {
                queue.add(child.getChild(index));
            }
        }
        return result;
    }

    /**
     * We use this to add new lines with appropriate indentations.
     *
     * @param text We will append the new lines and indentations into this.
     * @param newLines Number of new lines before appending indentations.
     */
    private void appendNewLinesAndIndent(final StringBuilder text, final int newLines) {
        text.append("\n".repeat(newLines))
            .append(INDENT_UNIT.repeat(this.currentIndentLevel));
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = "[one, two, three] [apple, banana, [one, two]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.file();
        final GrammarVisitor visitor = new GrammarVisitor();
        final String result = visitor.visit(tree);
        System.out.println(result);
    }

}
```

Here is the output:
```
[one, two, three]
[
  apple,
  banana,
  [one, two]
]
```
