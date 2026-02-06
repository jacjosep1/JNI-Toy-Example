
# JNI Toy Example
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

# Incidence Data Structure
## Installation
Requires Python 3.12.3 or similar.  
```
python -m venv venv
pip install -r requirements.txt
```