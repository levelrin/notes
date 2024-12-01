## Goal

Take a look at this input: `2 9 10 3 1 2 3`.

The first integer, `2`, determines the number of integers in the first group.

So, the first group has the following numbers: `[9, 10]`.

Likewise, the next integer, `3`, determines the number of integers in the next group.

So, the second group has the following numbers: `[1, 2, 3]`.

As you can see, the grouping rule depends on the situation.

To deal with this kind of problem, we can use parameterized rules.

## Grammar

```g4
grammar Grammar;

@header {package com.levelrin.antlr.generated;}

file
    : group+
    ;

group
    // the parser rule `sequence` takes a parameter.
    // `$INT.int` means the group INT's integer value.
    : INT sequence[$INT.int]
    ;

// This rule has an integer parameter n.
sequence[int n]
// Initialize the local variable i to 0.
locals [int i = 0;]
    // Match the integer n times.
    // {}? is a conditional statement.
    // The rule continues only if the condition is met.
    // After mathcing INT, increment i by 1.
    : ( {$i < $n}? INT {$i++;} )*
    ;

INT: [0-9]+;
WS: [ \t\r\n]+ -> skip;
```

By the way, I noticed that IntelliJ's Antlr4 plugin does not support parameterized rules.

For that reason, I had to convert the AST into XML format and print it to see the parsing result.

## Java

```java
package com.levelrin.antlr4test;

import com.levelrin.antlr.generated.GrammarBaseListener;
import com.levelrin.antlr.generated.GrammarLexer;
import com.levelrin.antlr.generated.GrammarParser;
import java.io.StringWriter;
import java.util.Locale;
import java.util.Objects;
import java.util.StringJoiner;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

/**
 * Note that the class name {@link com.levelrin.antlr.generated.GrammarListener} already exists.
 */
class OurGrammarListener extends GrammarBaseListener {

    /**
     * This listener will put AST information into this XML object.
     */
    private final Document xml;

    private Node currentRule;

    /**
     * Constructor.
     * @param xml Please pass an empty XML object.
     */
    public OurGrammarListener(final Document xml) {
        this.xml = xml;
    }

    @Override
    public void enterEveryRule(final ParserRuleContext context) {
        // It has `Context` suffix.
        final String simpleClassName = context.getClass().getSimpleName();
        // The rule name written in the grammar file.
        final String ruleName = String.format(
            "%s%s",
            // Make the first character lowercase.
            simpleClassName.substring(0, 1).toLowerCase(Locale.ROOT),
            // -7 is for removing the `Context` suffix.
            simpleClassName.substring(1, simpleClassName.length() - 7)
        );
        final Element rule = this.xml.createElement(ruleName);
        // We will add the rule's terminals using this joiner.
        final StringJoiner terminals = new StringJoiner(" ");
        rule.setUserData("terminals", terminals, (operation, key, data, src, dst) -> {});
        Objects.requireNonNullElse(this.currentRule, this.xml).appendChild(rule);
        this.currentRule = rule;
    }

    @Override
    public void exitEveryRule(final ParserRuleContext context) {
        this.currentRule = this.currentRule.getParentNode();
    }

    @Override
    public void visitTerminal(final TerminalNode node) {
        if (this.currentRule == null) {
            throw new IllegalStateException("A terminal node without rule detected. Please start with a rule.");
        } else {
            final StringJoiner terminals = (StringJoiner) this.currentRule.getUserData("terminals");
            terminals.add(node.getText());
            // This assumes that the current rule only has the text element.
            // In other words, we assume that there is no rule that has rules AND texts.
            this.currentRule.setTextContent(terminals.toString());
        }
    }
}

public class Main {

    public static void main(String... args) throws Exception {
        final String originalText = "2 9 10 3 1 2 3";
        final GrammarParser parser = new GrammarParser(
            new CommonTokenStream(
                new GrammarLexer(
                    CharStreams.fromString(originalText)
                )
            )
        );
        final ParseTree tree = parser.file();
        final Document xml = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        final OurGrammarListener listener = new OurGrammarListener(xml);
        ParseTreeWalker.DEFAULT.walk(listener, tree);
        final Transformer transformer = TransformerFactory.newInstance().newTransformer();
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
        final StringWriter writer = new StringWriter();
        transformer.transform(new DOMSource(xml), new StreamResult(writer));
        final String text = writer.toString();
        System.out.println(text);
    }

}
```

Output:
```xml
<file>
  <group>
    2
    <sequence>9 10</sequence>
  </group>
  <group>
    3
    <sequence>1 2 3</sequence>
  </group>
</file>
```
