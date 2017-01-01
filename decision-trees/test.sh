#!/bin/bash

mkdir bin

make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the featurre files may be built. Note that don't necessarily have to
# use Java for this step.

#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.all ./../badges.example.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator1 ../badges/badges.modified.data.all ./../badges.example.arff

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1 ./../badges.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2 ./../badges.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold3 ./../badges.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold4 ./../badges.fold4.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold5 ./../badges.fold5.arff

# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation.

#WekarTester(given)
java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.example.arff

#SGD StumpTester
java -cp lib/weka.jar:bin cs446.homework2.StumpTester ./../badges.example.arff ./../stumps.arff

#SGD Tester
java -cp lib/weka.jar:bin cs446.homework2.SGDTester ./../trainAndTest_arff/training1.arff ./../trainAndTest_arff/testing1.arff ./../trainAndTest_arff/training2.arff ./../trainAndTest_arff/testing2.arff ./../trainAndTest_arff/training3.arff ./../trainAndTest_arff/testing3.arff ./../trainAndTest_arff/training4.arff ./../trainAndTest_arff/testing4.arff ./../trainAndTest_arff/training5.arff ./../trainAndTest_arff/testing5.arff

#ID3(4) Tester
java -cp lib/weka.jar:bin cs446.homework2.ID3depth4Tester ./../trainAndTest_arff/training1.arff ./../trainAndTest_arff/testing1.arff ./../trainAndTest_arff/training2.arff ./../trainAndTest_arff/testing2.arff ./../trainAndTest_arff/training3.arff ./../trainAndTest_arff/testing3.arff ./../trainAndTest_arff/training4.arff ./../trainAndTest_arff/testing4.arff ./../trainAndTest_arff/training5.arff ./../trainAndTest_arff/testing5.arff

#ID3(8) Tester
java -cp lib/weka.jar:bin cs446.homework2.ID3depth8Tester ./../trainAndTest_arff/training1.arff ./../trainAndTest_arff/testing1.arff ./../trainAndTest_arff/training2.arff ./../trainAndTest_arff/testing2.arff ./../trainAndTest_arff/training3.arff ./../trainAndTest_arff/testing3.arff ./../trainAndTest_arff/training4.arff ./../trainAndTest_arff/testing4.arff ./../trainAndTest_arff/training5.arff ./../trainAndTest_arff/testing5.arff

#ID3(full) Tester
java -cp lib/weka.jar:bin cs446.homework2.ID3depthfullTester ./../trainAndTest_arff/training1.arff ./../trainAndTest_arff/testing1.arff ./../trainAndTest_arff/training2.arff ./../trainAndTest_arff/testing2.arff ./../trainAndTest_arff/training3.arff ./../trainAndTest_arff/testing3.arff ./../trainAndTest_arff/training4.arff ./../trainAndTest_arff/testing4.arff ./../trainAndTest_arff/training5.arff ./../trainAndTest_arff/testing5.arff




#### following testers use original features ####
#### comment commands above and uncomment following commands to test on original features. ALSO use FeatureGenerator1.java at the beginning ####




#java -cp lib/weka.jar:bin cs446.homework2.SGDTester ./../trainAndTest_arff1/training1.arff ./../trainAndTest_arff1/testing1.arff ./../trainAndTest_arff1/training2.arff ./../trainAndTest_arff1/testing2.arff ./../trainAndTest_arff1/training3.arff ./../trainAndTest_arff1/testing3.arff ./../trainAndTest_arff1/training4.arff ./../trainAndTest_arff1/testing4.arff ./../trainAndTest_arff1/training5.arff ./../trainAndTest_arff1/testing5.arff

#java -cp lib/weka.jar:bin cs446.homework2.ID3depth4Tester ./../trainAndTest_arff1/training1.arff ./../trainAndTest_arff1/testing1.arff ./../trainAndTest_arff1/training2.arff ./../trainAndTest_arff1/testing2.arff ./../trainAndTest_arff1/training3.arff ./../trainAndTest_arff1/testing3.arff ./../trainAndTest_arff1/training4.arff ./../trainAndTest_arff1/testing4.arff ./../trainAndTest_arff1/training5.arff ./../trainAndTest_arff1/testing5.arff

#java -cp lib/weka.jar:bin cs446.homework2.ID3depth8Tester ./../trainAndTest_arff1/training1.arff ./../trainAndTest_arff1/testing1.arff ./../trainAndTest_arff1/training2.arff ./../trainAndTest_arff1/testing2.arff ./../trainAndTest_arff1/training3.arff ./../trainAndTest_arff1/testing3.arff ./../trainAndTest_arff1/training4.arff ./../trainAndTest_arff1/testing4.arff ./../trainAndTest_arff1/training5.arff ./../trainAndTest_arff1/testing5.arff

#java -cp lib/weka.jar:bin cs446.homework2.ID3depthfullTester ./../trainAndTest_arff1/training1.arff ./../trainAndTest_arff1/testing1.arff ./../trainAndTest_arff1/training2.arff ./../trainAndTest_arff1/testing2.arff ./../trainAndTest_arff1/training3.arff ./../trainAndTest_arff1/testing3.arff ./../trainAndTest_arff1/training4.arff ./../trainAndTest_arff1/testing4.arff ./../trainAndTest_arff1/training5.arff ./../trainAndTest_arff1/testing5.arff


