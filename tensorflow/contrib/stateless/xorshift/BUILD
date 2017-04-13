# Random access into xorshift128+

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

cc_library(
    name = "xorshift",
    srcs = [
        "xorshift_generated.h",
        "xorshift.cc",
    ],
    hdrs = [
        "xorshift.h",
    ],
    copts = [
        "-mavx2",
        "-mpclmul",
    ],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "xorshift_test",
    srcs = ["xorshift_test.cc"],
    deps = [
        ":xorshift",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "gen_xorshift",
    srcs = [
        "gen_xorshift.cc",
        "xorshift.h",
    ],
    deps = [
        "//tensorflow/core:test",
    ],
)
