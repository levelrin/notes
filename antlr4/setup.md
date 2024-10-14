## Antlr4 Gradle Plugin Setup

Here is the configuration for the `build.gradle` file:
```groovy
plugins {
    // ...
    id 'antlr'
}

sourceSets {
    main {
        java {
            // We will generate the parser in this directory using Antlr.
            // We need to add this directory to the main source set to use the generated classes in our code.
            srcDirs("$buildDir/generated-src/antlr/main")
        }
    }
}

generateGrammarSource {
    maxHeapSize = '64m'
    arguments += [
        '-visitor',
        '-long-messages'
    ]
    // Since we will add the package declaration, we must put generated classes in the package folder.
    outputDirectory = file("$buildDir/generated-src/antlr/main/com/levelrin/antlr/generated")
}

dependencies {
    antlr 'org.antlr:antlr4:4.13.1'
    // ...
}
```

Next, create a new directory `src/main/antlr`.

The grammar files would go in that directory like this: `src/main/antlr/Grammar.g4`.

Note that we have to declare the header for the package statements like this:
```g4
grammar Grammar;

// This will create package statements for the generated Java files.
@header {package com.levelrin.antlr.generated;}

greeting
    : HELLO WORLD
    ;

HELLO: 'Hello';
WORLD: 'World';
WS: [ \t\r\n]+ -> skip;
```

Finally, we can generate the Java classes using this command: `./gradlew generateGrammarSource`.
