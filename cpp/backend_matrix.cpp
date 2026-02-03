#include <jni.h>
#include <iostream>
#include <vector>

struct csr_matrix {
    std::vector<int> indptr{};
    std::vector<int> indices{};
    std::vector<double> data{};

    struct vector_data {
        size_t size;
        void *data;
    };
    template<class T>
    vector_data construct(std::vector<T> &_v) {
        return vector_data{_v.size(), reinterpret_cast<void*>(_v.data())};
    }

    struct buffer_data {
        vector_data indptr;
        vector_data indices;
        vector_data data;
    };
    inline buffer_data buffer() {
        return buffer_data{
            construct(indptr),
            construct(indices),
            construct(data)
        };
    }
};

static csr_matrix TEST_MATRIX{
    {0, 2, 3, 6},           // indprt
    {0, 3, 1, 0, 2, 3},     // indices
    {1, 2, 3, 4, 5, 6}      // data
};

extern "C" {

JNIEXPORT jobject JNICALL
Java_Backend_link_matrix(JNIEnv* env, jobject, jint n) {
    return env->NewDirectByteBuffer(
        TEST_MATRIX.buffer(),
        TEST_MATRIX.buffer_size()
    );
}

}
