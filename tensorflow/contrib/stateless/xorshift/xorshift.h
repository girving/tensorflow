/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Random access into xorshift128+

// Following Vigna's https://arxiv.org/abs/1404.0390, a fixed size jump in
// xorshift128+ can be computed by precomputed a GF(2) polynomial modulo
// the characteristic polynomial of one step of xorshift128+ (which is
// linear as a transform on GF(2)^128).
//
// We can jump by a variable amount by precomputing these polynomials for
// each power of two jump, then multiplying them all together.
// Unfortunately, multiplication of polynomials modulo the characteristic
// polynomial is slow, since that modulus is dense.  Fortunately, all
// finite fields of a given order are isomorphic, so we can do all the
// multiplications in the equivalent field modulo the pentanomial
// x^128 + x^7 + x^2 + x + 1.  This is reasonably fast if we take advantage
// of the carryless multiply instructions on recent CPUs.  We then convert
// back to the slow modulus using a precomputed linear map corresponding to
// the finite field isomorphism.
//
// If the base point is fixed, we can compose the GF(2^128) -> GF(2^128)
// linear map with the linear map corresponding to the final jump, for some
// extra speedup.
//
// The result is that we can use xorshift128+ as a reasonably efficient
// counter mode generator: seeking into the appropriate part of the stream
// and using sequential mode from there.
//
// Caveat: This code is extremely nonportable.  If it's ever to actually be
// used, slower versions for less capable hardware will have to be written,
// ideally using runtime detection of which instructions exist.

#ifndef TENSORFLOW_CONTRIB_STATELESS_XORSHIFT_XORSHIFT_H_
#define TENSORFLOW_CONTRIB_STATELESS_XORSHIFT_XORSHIFT_H_

#include <array>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using uint128 = __uint128_t;

// One step of xorshift128+
inline std::array<uint64, 2> XorshiftStep(const std::array<uint64, 2> state) {
  auto x = state[0];
  const auto y = state[1];
  x ^= x << 23;
  return {{y, x ^ y ^ (x >> 17) ^ (y >> 26)}};
}

// Many steps of xorshift128+.  XorshiftJump(s, n) is equivalent to n calls
// to XorshiftStep on s, but is constant time.
std::array<uint64, 2> XorshiftJump(std::array<uint64, 2> state,
                                   const uint128 steps);

// Starting point for XorshiftBaseJump
constexpr std::array<uint64, 2> kXorshiftJumpBase = {{
    0x44d4a1ed567714f5, 0xc6f60fef2dd13f8b
}};

// XorshiftBaseJump(n) = XorshiftJump(kXorshiftJumpBase, n), but is faster since
// it allows more precomputation.
std::array<uint64, 2> XorshiftBaseJump(const uint128 steps);

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_STATELESS_XORSHIFT_XORSHIFT_H_
