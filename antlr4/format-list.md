## Goal

Before:
```
[['a','b'],['c','d'],[['e','f'],['g','h'],[]]]
```

After:
```
[
  ['a', 'b', ],
  ['c', 'd', ],
  [
    ['e', 'f', ],
    ['g', 'h', ],
    [],
  ],
]
```

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

listDeclaration
    : '[' listValue (',' listValue)*']'
    ;

listValue
    : listDeclaration
    | listEmpty
    | listSingleLiteral
    ;

listEmpty
    : '[' ']'
    ;

listSingleLiteral
    : ELEMENT
    ;

ELEMENT: '\'' [a-zA-Z0-9]* '\'';
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
import org.antlr.v4.runtime.tree.ParseTree;

final class GrammarVisitor extends GrammarBaseVisitor<String> {

    private int indentLevel = 0;

    private int listNum = 0;

    @Override
    public String visitListDeclaration(final GrammarParser.ListDeclarationContext context) {
        final StringBuilder result = new StringBuilder();
        result.append("[");
        final int listNumBefore = this.listNum;
        for (final GrammarParser.ListValueContext element : context.listValue()) {
            this.visit(element);
        }
        final int listNumAfter = this.listNum;
        final boolean hasListElement = listNumBefore != listNumAfter;
        if (hasListElement) {
            this.indentLevel++;
        }
        for (final GrammarParser.ListValueContext element : context.listValue()) {
            if (hasListElement) {
                result
                    .append("\n")
                    .append("  ".repeat(this.indentLevel))
                    .append(this.visit(element))
                    .append(",");
            } else {
                result
                    .append(this.visit(element))
                    .append(", ");
            }
        }
        if (hasListElement) {
            this.indentLevel--;
            result.append("\n").append("  ".repeat(this.indentLevel)).append("]");
        } else {
            result.append("]");
        }
        return result.toString();
    }

    @Override
    public String visitListValue(final GrammarParser.ListValueContext context) {
        if (context.listDeclaration() != null) {
            this.listNum++;
        }
        return this.visit(context.getChild(0));
    }

    @Override
    public String visitListEmpty(final GrammarParser.ListEmptyContext context) {
        return "[]";
    }

    @Override
    public String visitListSingleLiteral(final GrammarParser.ListSingleLiteralContext context) {
        return context.ELEMENT().getText();
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
