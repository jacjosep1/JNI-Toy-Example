#include <jni.h>
#include <iostream>

static float* g_buf = nullptr;
static int g_n = 0;

extern "C" {

JNIEXPORT jobject JNICALL
Java_Backend_allocate(JNIEnv* env, jobject, jint n) {
    g_n = n;

    delete[] g_buf;
    g_buf = new float[n];

    return env->NewDirectByteBuffer(
        (void*)g_buf,
        n * sizeof(float)
    );
}

JNIEXPORT void JNICALL
Java_Backend_fill(JNIEnv*, jobject, jfloat value) {
    for (int i = 0; i < g_n; ++i) {
        g_buf[i] = value;
    }
}

}
