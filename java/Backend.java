
public class Backend {
    static {
        System.loadLibrary("backend");
    }

    public native java.nio.ByteBuffer allocate(int n);
    public native void fill(float value);
    public native CsrMatrix linkMatrix();
}
