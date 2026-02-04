import java.nio.*;

public class CsrMatrix {
    public final IntBuffer indptr;
    public final IntBuffer indices;
    public final DoubleBuffer data;

    public CsrMatrix(ByteBuffer i, ByteBuffer j, ByteBuffer d) {
        indptr  = i.order(ByteOrder.nativeOrder()).asIntBuffer();
        indices = j.order(ByteOrder.nativeOrder()).asIntBuffer();
        data    = d.order(ByteOrder.nativeOrder()).asDoubleBuffer();
    }
}
