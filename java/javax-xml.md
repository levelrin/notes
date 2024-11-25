## Parse

```java
package com.levelrin;

import java.io.ByteArrayInputStream;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathFactory;
import org.w3c.dom.Document;

public final class Main {

    public static void main(final String... args) throws Exception {
        final String xml = """
            <root>
                <person>
                    <name>John</name>
                    <age>30</age>
                </person>
                <person>
                    <name>Jane</name>
                    <age>25</age>
                </person>
            </root>
            """;
        // Step 1: Parse the XML string into a Document.
        final DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        final DocumentBuilder builder = factory.newDocumentBuilder();
        final Document document = builder.parse(new ByteArrayInputStream(xml.getBytes()));

        // Step 2: Create an XPath instance.
        final XPathFactory xPathFactory = XPathFactory.newInstance();
        final XPath xPath = xPathFactory.newXPath();

        // Step 3: Compile an XPath expression.
        final String expression = "/root/person[name='Jane']/age";
        final XPathExpression xPathExpression = xPath.compile(expression);

        // Step 4: Evaluate the expression.
        final String age = (String) xPathExpression.evaluate(document, XPathConstants.STRING);

        // Output the result.
        System.out.println("Jane's age: " + age);
    }

}
```
