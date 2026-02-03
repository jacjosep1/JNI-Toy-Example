#!/bin/bash
set -e

mkdir -p build

# compile Java
javac -d build java/*.java

# generate JNI headers
javac -h cpp java/Backend.java

# compile shared lib
g++ -O3 -fPIC \
    -I"$JAVA_HOME/include" \
    -I"$JAVA_HOME/include/linux" \
    -shared cpp/backend.cpp \
    -o build/libbackend.so

# run
java -Djava.library.path=build -cp build Main
