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

        // CSR matrix test
        CsrMatrix m1 = b.linkMatrix();

        System.out.println(m1.indptr.get(0));
        System.out.println(m1.data.get(0));

        CsrMatrix m2 = b.linkMatrix();
        m2.indptr.put(0, 7);
        m2.data.put(0, 8);

        // Print m1 again to verify we are sharing the same data - should print 7, 8
        System.out.println(m1.indptr.get(0));
        System.out.println(m1.data.get(0));
    }
}
