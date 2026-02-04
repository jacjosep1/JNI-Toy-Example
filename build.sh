#!/bin/bash
set -e

mkdir -p build

# auto-detect JAVA_HOME from javac location
JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))

# compile Java
javac -d build java/*.java

# generate JNI headers
javac -h cpp java/Backend.java java/CsrMatrix.java

# compile shared lib
g++ -O3 -fPIC \
  -I"$JAVA_HOME/include" \
  -I"$JAVA_HOME/include/linux" \
  -shared cpp/*.cpp \
  -o build/libbackend.so

# run
java -Djava.library.path=build -cp build Main
