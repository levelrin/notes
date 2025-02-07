# Goal

Before:
```
[['a','b'],['c','d'],[['e','f'],['g','h'],[]]]
```

After:
```
[
  ['a', 'b'],
  ['c', 'd'],
  [
    ['e', 'f'],
    ['g', 'h'],
    []
  ]
]
```

# Visitor Solution

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

listDeclaration
    : OPENING_BRACKET listItems CLOSING_BRACKET
    | emptyList
    ;

listItems
    : listItem (COMMA listItem)*
    ;

listItem
    : STRING
    | listDeclaration
    ;

emptyList
    : OPENING_BRACKET CLOSING_BRACKET
    ;

OPENING_BRACKET: '[';
CLOSING_BRACKET: ']';
COMMA: ',';
STRING: '\'' ~[\r\n']* '\'';
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
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;

class GrammarVisitor extends GrammarBaseVisitor<String> {

    private static final String INDENT_UNIT = "  ";

    private int currentIndentLevel;

    @Override
    public String visitListDeclaration(final GrammarParser.ListDeclarationContext context) {
        final TerminalNode openingBracket = context.OPENING_BRACKET();
        final GrammarParser.ListItemsContext listItems = context.listItems();
        final TerminalNode closingBracket = context.CLOSING_BRACKET();
        final GrammarParser.EmptyListContext emptyListContext = context.emptyList();
        final StringBuilder text = new StringBuilder();
        if (emptyListContext == null) {
            boolean nested = false;
            for (final GrammarParser.ListItemContext listItemContext : listItems.listItem()) {
                if (listItemContext.listDeclaration() != null) {
                    nested = true;
                }
            }
            if (nested) {
                text.append(this.visit(openingBracket))
                    .append('\n');
                this.currentIndentLevel++;
                text.append(INDENT_UNIT.repeat(this.currentIndentLevel))
                    .append(this.visitIndentedListItems(listItems))
                    .append('\n');
                this.currentIndentLevel--;
                text.append(INDENT_UNIT.repeat(this.currentIndentLevel))
                    .append(this.visit(closingBracket));
            } else {
                text.append(this.visit(openingBracket))
                    .append(this.visit(listItems))
                    .append(this.visit(closingBracket));
            }
        } else {
            text.append(this.visit(emptyListContext));
        }
        return text.toString();
    }

    @Override
    public String visitListItems(final GrammarParser.ListItemsContext context) {
        final List<GrammarParser.ListItemContext> listItemContexts = context.listItem();
        final List<TerminalNode> commaTerminals = context.COMMA();
        final StringBuilder text = new StringBuilder();
        final GrammarParser.ListItemContext firstListItemContext = listItemContexts.get(0);
        text.append(this.visit(firstListItemContext));
        for (int index = 0; index < commaTerminals.size(); index++) {
            final TerminalNode commaTerminal = commaTerminals.get(index);
            final GrammarParser.ListItemContext listItemContext = listItemContexts.get(index + 1);
            text.append(this.visit(commaTerminal))
                .append(' ')
                .append(this.visit(listItemContext));
        }
        return text.toString();
    }

    public String visitIndentedListItems(final GrammarParser.ListItemsContext context) {
        final List<GrammarParser.ListItemContext> listItemContexts = context.listItem();
        final List<TerminalNode> commaTerminals = context.COMMA();
        final StringBuilder text = new StringBuilder();
        final GrammarParser.ListItemContext firstListItemContext = listItemContexts.get(0);
        text.append(this.visit(firstListItemContext));
        for (int index = 0; index < commaTerminals.size(); index++) {
            final TerminalNode commaTerminal = commaTerminals.get(index);
            final GrammarParser.ListItemContext listItemContext = listItemContexts.get(index + 1);
            text.append(this.visit(commaTerminal))
                .append('\n')
                .append(INDENT_UNIT.repeat(this.currentIndentLevel))
                .append(this.visit(listItemContext));
        }
        return text.toString();
    }

    @Override
    public String visitListItem(final GrammarParser.ListItemContext context) {
        final TerminalNode stringTerminal = context.STRING();
        final GrammarParser.ListDeclarationContext listDeclaration = context.listDeclaration();
        final StringBuilder text = new StringBuilder();
        if (stringTerminal == null) {
            text.append(this.visit(listDeclaration));
        } else {
            text.append(this.visit(stringTerminal));
        }
        return text.toString();
    }

    @Override
    public String visitEmptyList(final GrammarParser.EmptyListContext context) {
        final TerminalNode openingBracketTerminal = context.OPENING_BRACKET();
        final TerminalNode closingBracketTerminal = context.CLOSING_BRACKET();
        final StringBuilder text = new StringBuilder();
        text.append(this.visit(openingBracketTerminal))
            .append(this.visit(closingBracketTerminal));
        return text.toString();
    }

    @Override
    public String visitTerminal(final TerminalNode node) {
        return node.getText();
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = "[['a','b'],['c','d'],[['e','f'],['g','h'],[]]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.listDeclaration();
        final GrammarVisitor visitor = new GrammarVisitor();
        final String result = visitor.visit(tree);
        System.out.printf("Before:%n%s%n%nAfter:%n%s%n", originalText, result);
    }

}
```

# Listener Solution

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

listDeclaration
    : OPENING_BRACKET listItems CLOSING_BRACKET
    | emptyList
    ;

listItems
    : listItem (itemSeperator listItem)*
    ;

itemSeperator
    : COMMA
    ;

listItem
    : STRING
    | listDeclaration
    ;

emptyList
    : OPENING_BRACKET CLOSING_BRACKET
    ;

OPENING_BRACKET: '[';
CLOSING_BRACKET: ']';
COMMA: ',';
STRING: '\'' ~[\r\n']* '\'';
WS: [ \t\r\n]+ -> skip;
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
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarListener extends GrammarBaseListener {

    private static final String INDENT_UNIT = "  ";

    private int currentIndentLevel;

    private final StringBuilder output;

    private final List<Integer> nestedStartTokenIndexes = new ArrayList<>();

    public GrammarListener(final StringBuilder output) {
        this.output = output;
    }

    @Override
    public void enterListDeclaration(final GrammarParser.ListDeclarationContext context) {
        final GrammarParser.ListItemsContext listItems = context.listItems();
        if (listItems != null) {
            boolean nested = false;
            for (final GrammarParser.ListItemContext listItemContext : listItems.listItem()) {
                if (listItemContext.listDeclaration() != null) {
                    nested = true;
                }
            }
            if (nested) {
                this.nestedStartTokenIndexes.add(context.getStart().getTokenIndex());
                this.output.append('[');
                this.currentIndentLevel++;
            } else {
                this.output.append('[');
            }
        } else {
            this.output.append('[');
        }
    }

    @Override
    public void enterListItem(final GrammarParser.ListItemContext context) {
        if (this.nestedStartTokenIndexes.contains(context.getParent().getParent().getStart().getTokenIndex())) {
            this.output.append('\n')
                .append(INDENT_UNIT.repeat(this.currentIndentLevel));
        }
        final TerminalNode stringTerminal = context.STRING();
        if (stringTerminal != null) {
            this.output.append(stringTerminal.getText());

        }
    }

    @Override
    public void enterItemSeperator(final GrammarParser.ItemSeperatorContext context) {
        if (this.nestedStartTokenIndexes.contains(context.getParent().getParent().getStart().getTokenIndex())) {
            this.output.append(',');
        } else {
            this.output.append(", ");
        }
    }

    @Override
    public void exitListDeclaration(final GrammarParser.ListDeclarationContext context) {
        if (this.nestedStartTokenIndexes.contains(context.getStart().getTokenIndex())) {
            this.output.append('\n');
            this.currentIndentLevel--;
            this.output.append(INDENT_UNIT.repeat(this.currentIndentLevel));
        }
        this.output.append(']');
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = "[['a','b'],['c','d'],[['e','f'],['g','h'],[]]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.listDeclaration();
        final StringBuilder output = new StringBuilder();
        final GrammarListener listener = new GrammarListener(output);
        ParseTreeWalker.DEFAULT.walk(listener, tree);
        System.out.printf("Before:%n%s%n%nAfter:%n%s%n", originalText, output);
    }

}
```
