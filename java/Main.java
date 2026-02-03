import java.nio.*;

public class Main {
    public static void main(String[] args) {
        Backend b = new Backend();

        int N = 8;

        // IMPORTANT: nativeOrder() for correct float interpretation
        ByteBuffer bb = b.allocate(N).order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();

        // write from Java
        for (int i = 0; i < N; i++) {
            fb.put(i, 0.0f);
        }
        for (int i = 0; i < N; i++) {
            System.out.println(fb.get(i));
        }

        // overwrite from C++
        b.fill(1.0f);

        // read again
        for (int i = 0; i < N; i++) {
            System.out.println(fb.get(i));
        }
    }
}
