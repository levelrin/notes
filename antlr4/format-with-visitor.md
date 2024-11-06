## Goal

We want to format a list like the following (adding a space between each element):
```
Before: [['a'],'b',['c','d'],[[]]]
After:  [['a'], 'b', ['c', 'd'], [[]]]
```

## Grammar

```g4
// It must match the file name.
grammar Grammar;

// This will create package statements for the generated Java files.
@header {package com.levelrin.antlr.generated;}

// Parser rules start with a lowercase letter.
// Tips:
//  - Do not mix lexer and parser rules because we need to figure out which one it is.
//  - Create a parser rule if we want to manipulate.
//  - Do not use countable (?, +, or *) if there are different parser rules because we need to check their types when we manipulate them.

// Although it's not a good idea to mix literals and parser rules, this case is okay
// because we are not interested in distinguishing literals and parser rules in this case.
// Also, there is only one parser rule, so we don't have to do any type checking.
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

// Lexer rules start with a capital letter.
ELEMENT: '\'' [a-zA-Z0-9]* '\'';
WS: [ \t\r\n]+ -> skip;
```

## Java

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseVisitor;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import java.util.StringJoiner;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;

final class GrammarVisitor extends GrammarBaseVisitor<String> {

    @Override
    public String visitListDeclaration(final GrammarParser.ListDeclarationContext context) {
        // Put a space between each element in the list.
        final StringJoiner elements = new StringJoiner(", ");
        for (final GrammarParser.ListValueContext element : context.listValue()) {
            elements.add(this.visit(element));
        }
        return String.format("[%s]", elements);
    }

    @Override
    public String visitListValue(final GrammarParser.ListValueContext context) {
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
        final String originalText = "[['a'],'b',['c','d'],[[]]]";
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        // Call the root-level parser rule.
        final ParseTree tree = parser.listDeclaration();
        final GrammarVisitor visitor = new GrammarVisitor();
        final String result = visitor.visit(tree);
        System.out.printf("Before: %s%nAfter:  %s%n", originalText, result);
    }

}
```
