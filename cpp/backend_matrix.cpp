#include <jni.h>
#include <iostream>
#include <vector>

struct csr_matrix {
    typedef int CSR_INDEXING_T;
    typedef double CSR_DTYPE_T;

    std::vector<CSR_INDEXING_T> indptr{};
    std::vector<CSR_INDEXING_T> indices{};
    std::vector<CSR_DTYPE_T> data{};

    jobject to_java_object(JNIEnv* env) {
        jclass cls = env->FindClass(_JAVA_CLASS);
        jmethodID ctor = env->GetMethodID(cls, "<init>", _JAVA_SIGNATURE);
        return env->NewObject(cls, ctor, 
            construct(indptr).java_byte_buffer(env), 
            construct(indices).java_byte_buffer(env), 
            construct(data).java_byte_buffer(env)
        );
    }

private:
    struct vector_data_t {
        size_t size;    // size in bytes
        void *data;     // data pointer

        jobject java_byte_buffer(JNIEnv *env) {
            return env->NewDirectByteBuffer(data, size);
        }
    };

    template<class dtype>
    inline vector_data_t construct(std::vector<dtype> &_v) {
        return vector_data_t{_v.size() * sizeof(dtype), reinterpret_cast<void*>(_v.data())};
    }

    static constexpr char *_JAVA_CLASS =        "CsrMatrix";
    static constexpr char *_JAVA_SIGNATURE =    "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V";
};

static csr_matrix TEST_MATRIX{
    {0, 2, 3, 6},           // indprt
    {0, 3, 1, 0, 2, 3},     // indices
    {1, 2, 3, 4, 5, 6}      // data
};

extern "C" {

JNIEXPORT jobject JNICALL
Java_Backend_linkMatrix(JNIEnv *env, jobject) {
    return TEST_MATRIX.to_java_object(env);
}

}
