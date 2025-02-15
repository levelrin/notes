## About

We can use `lexical modes` if we want to apply a different set of lexer rules.

A common use case is when we have a language inside of another language.

For example, we may need to apply different lexer rules for JavaDoc that are separate from the regular Java grammar.

By the way, a grammar for an inner language, such as JavaDoc, is called an `island grammar.`

## Goal

Let's say we want to parse a string interpolation like this: `outer.method1("one ${inner.method2("two")} three");`.

The tricky part is that we have a method call as a part of the string literal.

## Grammar

For this example, we prepare two grammar files.

`GrammarLexer.g4`:
```g4
lexer grammar GrammarLexer;

@header {package com.levelrin.antlr.generated;}

// Lexer rules for the `DEFAULT_MODE`.
// As the name suggests, it's the predefined mode by Antlr4.
DOT: '.';
OPEN_PARENTHESIS: '(';
CLOSE_PARENTHESIS: ')';
SEMICOLON: ';';
NAME: [a-z]([a-z0-9-]*[a-z0-9])?;
// Switch the mode to `STRING`.
STRING_START: '"' -> pushMode(STRING);
INTERPOLATION_END: '}' -> pushMode(STRING);
WS: [ \r\n] -> skip;

// From here, we define lexer rules for the mode `STRING`.
// Note that defining a mode is only allowed in the lexer grammar file.
// That's why we created a separate file for lexer rules.
mode STRING;
// Switch the mode back to the previous mode.
// In this case, it will be `DEFAULT_MODE`.
// FYI, we will encounter `java.util.EmptyStackException` if there is nothing to pop.
INTERPOLATION_START: '${' -> popMode;
// All characters except `"` or `$`.
// We actually need to exclude the sequence of characters `${` instead of `$`.
// However, Antlr4 does not support the negation of a sequence.
// A workaround is outside of this study's scope.
STRING_LITERAL: ~["$]+;
STRING_END: '"' -> popMode;
```

`GrammarParser.g4`:
```g4
parser grammar GrammarParser;

@header {package com.levelrin.antlr.generated;}

// This is how we can import lexer rules.
// Note that we need to generate the grammar source first.
// It will generate the `GrammarLexer.tokens` file.
// We need to place that file in the same directory as this parser grammar file.
// If we update the lexer rules, we need to generate the source and place the tokens file again.
options {
    tokenVocab = GrammarLexer;
}

functionCall
    : NAME DOT NAME OPEN_PARENTHESIS string CLOSE_PARENTHESIS SEMICOLON?
    ;

string
    : STRING_START stringComponent* STRING_END
    ;

stringComponent
    : STRING_LITERAL
    | interpolation
    ;

interpolation
    : INTERPOLATION_START functionCall INTERPOLATION_END
    ;
```

## Java

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import com.levelrin.antlr.generated.GrammarParserBaseListener;
import java.util.List;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.antlr.v4.runtime.tree.TerminalNode;

final class GrammarParserListener  extends GrammarParserBaseListener {

    @Override
    public void enterFunctionCall(final GrammarParser.FunctionCallContext context) {
        final List<TerminalNode> nameTerminals = context.NAME();
        final TerminalNode objName = nameTerminals.get(0);
        final TerminalNode methodName = nameTerminals.get(1);
        System.out.println(objName.getText() + "#" + methodName.getText());
    }

}

public class Main {

    public static void main(String... args) {
        final String originalText = """
            outer.method1("one ${inner.method2("two")} three");
            """;
        final CharStream charStream = CharStreams.fromString(originalText);
        final GrammarLexer lexer = new GrammarLexer(charStream);
        final CommonTokenStream tokens = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokens);
        final ParseTree tree = parser.functionCall();
        final GrammarParserListener listener = new GrammarParserListener();
        ParseTreeWalker.DEFAULT.walk(listener, tree);
    }

}
```
