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

#ifndef TENSORFLOW_CONTRIB_STATELESS_XORSHIFT_GF1P128_H_
#define TENSORFLOW_CONTRIB_STATELESS_XORSHIFT_GF1P128_H_

#include <emmintrin.h>
#include <immintrin.h>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// GF(2^128) represented modulo x^128 + x^7 + x^2 + x + 1.
class GF1p128 {
 public:
  explicit GF1p128(__m128i c) : c_(c) {}
  GF1p128(uint64 lo, uint64 hi) : c_(_mm_set_epi64x(hi, lo)) {}

  __m128i c() const { return c_; }
  GF1p128 operator+(const GF1p128 b) const { return GF1p128(c_ ^ b.c_); }

  bool operator==(const GF1p128 b) const {
    return _mm_movemask_epi8(
        ~_mm_cmpeq_epi32(c_ ^ b.c_, _mm_setzero_si128())) == 0;
  }

 private:
  __m128i c_;
};

// Swap low and high 64-bit chunks of an __m128i
inline __m128i swap(const __m128i x) {
  return _mm_shuffle_epi32(x, _MM_SHUFFLE(1,0,3,2));
}

// Shift each 64-bit chunk left.  Use a macro to ensure n is an immediate.
template <int shift> inline __m128i left_shift(const __m128i x) {
  return _mm_slli_epi64(x, shift) |
         _mm_slli_si128(_mm_srli_epi64(x, 64 - shift), 8);
}

// Multiply two elements of the GF(2^128) finite field.  For now, we assume
// availability of the special carryless multiply intrinsic _mm_clmulepi64_si128
// which Intel and AMD added to accelerate AES-GCM.  We also use the same
// irreducible polynomial modulus, even though this is not the modulus needed by
// xorshift128+.  Since all finite fields of a given order are isomorphism (here
// the order is 2^128), we can convert to the right modulus using a linear
// transform when necessary.
inline GF1p128 operator*(const GF1p128 a, const GF1p128 b) {
  // For details, see
  //   Intel's 1230-Carry-Less-Multiplication-and-The-GCM-Mode_WP_.pdf
  // In particular, we use the same letters for variables.
  const auto mask_lo = _mm_set_epi64x(0, ~static_cast<uint64>(0));
  const auto mask_hi = swap(mask_lo);

  // Multiply out to 256 bits, using Karatsuba to reduce 4 muls to 3.
  const auto c = _mm_clmulepi64_si128(a.c(), b.c(), 0x11);
  const auto d = _mm_clmulepi64_si128(a.c(), b.c(), 0x00);
  const auto e = _mm_clmulepi64_si128(
      a.c() ^ swap(a.c()), b.c() ^ swap(b.c()), 0x00);
  const auto cde = swap(c ^ d ^ e);
  const auto x01 = d ^ (mask_hi & cde);
  const auto x23 = c ^ (mask_lo & cde);

  // Reduce modulo x^128 + x^7 + x^2 + x + 1.
  const auto xd =
      (x23 & mask_hi) | (mask_lo & (x23 ^ swap(_mm_srli_epi64(x23, 63) ^
                                               _mm_srli_epi64(x23, 62) ^
                                               _mm_srli_epi64(x23, 57))));
  return GF1p128(
      x01 ^ xd ^ left_shift<1>(xd) ^ left_shift<2>(xd) ^ left_shift<7>(xd));
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_STATELESS_XORSHIFT_GF1P128_H_
