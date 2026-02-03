## Installation
Java installation
```
sudo apt install openjdk-21-jdk
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export PATH=$JAVA_HOME/bin:$PATH
```

Build
```
./build.sh
```