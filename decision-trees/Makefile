JFLAGS = -cp lib/weka.jar:bin -d bin -sourcepath src
JAVAC = javac

.PHONY: directories
.SUFFIXES: .java .class

.java.class:
	$(JAVAC) $(JFLAGS) $*.java

CLASSES = src/FeatureGenerator.java src/FeatureGenerator1.java src/Id3.java src/SGD.java src/WekaTester.java src/SGDTester.java src/ID3depth4Tester.java src/ID3depth8Tester.java src/ID3depthfullTester.java src/StumpTester.java

all: directories classes

directories: 
	mkdir -p bin

classes: $(CLASSES:.java=.class)

clean:
	$(RM) -r bin

