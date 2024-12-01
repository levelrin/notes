## Parse

```java
package com.levelrin;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathFactory;
import org.w3c.dom.Document;

public final class Main {

    public static void main(final String... args) throws Exception {
        final String raw = """
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

        // Create an XML object.
        final Document xml = DocumentBuilderFactory
            .newInstance()
            .newDocumentBuilder()
            .parse(
                new ByteArrayInputStream(
                    raw.getBytes(StandardCharsets.UTF_8)
                )
            );

        // Parse the XML using XPath.
        final XPath xPath = XPathFactory.newInstance().newXPath();
        final String age = (String) xPath
            .compile("/root/person[name='Jane']/age")
            .evaluate(xml, XPathConstants.STRING);
        System.out.println("Jane's age: " + age);
    }

}
```

## Stringify XML

```java
package com.levelrin;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.Document;

public final class Main {

    public static void main(final String... args) throws Exception {
        final String raw = """
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
        final Document xml = DocumentBuilderFactory
            .newInstance()
            .newDocumentBuilder()
            .parse(
                new ByteArrayInputStream(
                    raw.getBytes(StandardCharsets.UTF_8)
                )
            );

        // It's for getting the string representation of the XML object.
        final Transformer transformer = TransformerFactory.newInstance().newTransformer();
        // Enable indentation for formatting.
        // Note that it will have extra line breaks between each node.
        // If you don't want that, you need to remove all line breaks in the raw XML.
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        // Set 2 spaces for each indentation.
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        // The transformer automatically add XML declaration on the top.
        // It's for removing that declaration.
        transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");

        // Use the transformer to get the String.
        final StringWriter writer = new StringWriter();
        transformer.transform(new DOMSource(xml), new StreamResult(writer));
        final String text = writer.toString();
        System.out.println(text);
    }

}
```

## Add Nodes

```java
package com.levelrin;

import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public final class Main {

    public static void main(final String... args) throws Exception {
        final String raw = """
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

        // Create an XML object.
        final Document xml = DocumentBuilderFactory
            .newInstance()
            .newDocumentBuilder()
            .parse(
                new ByteArrayInputStream(
                    raw.getBytes(StandardCharsets.UTF_8)
                )
            );

        // Get XML node objects for John and Jane.
        final XPath xPath = XPathFactory.newInstance().newXPath();
        final Node john = (Node) xPath
            .compile("/root/person[name='John']")
            .evaluate(xml, XPathConstants.NODE);
        final Node jane = (Node) xPath
            .compile("/root/person[name='Jane']")
            .evaluate(xml, XPathConstants.NODE);

        // Add the dev position to Jane.
        final Element pmPosition = xml.createElement("position");
        // Note that this removes all the children.
        // Please use the `setTextContent` method only if you want to overwrite the entire children with the text.
        pmPosition.setTextContent("Project Manager");
        john.appendChild(pmPosition);

        // Add the PM position to John.
        final Element devPosition = xml.createElement("position");
        devPosition.setTextContent("Developer");
        jane.appendChild(devPosition);

        // Print the modified XML.
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
