public class Backend {
    static {
        System.loadLibrary("backend"); // loads libbackend.so / .dll
    }

    // allocate buffer in C++
    public native java.nio.ByteBuffer allocate(int n);

    // just to prove both sides see same memory
    public native void fill(float value);
}
