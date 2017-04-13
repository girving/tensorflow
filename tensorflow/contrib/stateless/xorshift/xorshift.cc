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

#include "tensorflow/contrib/stateless/xorshift/xorshift.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <wmmintrin.h>
#include "tensorflow/contrib/stateless/xorshift/xorshift_generated.h"
namespace tensorflow {

using std::array;

// Unpack an __m128i into low and high bit chunks
static inline array<uint64, 2> unpack(const __m128i x) {
  union {
    __m128i x;
    array<uint64, 2> s;
  } u;
  u.x = x;
  return u.s;
}

bool GF1p128::operator==(const GF1p128 b) const {
  return _mm_movemask_epi8(
      ~_mm_cmpeq_epi32(c_ ^ b.c_, _mm_setzero_si128())) == 0;
}

// Swap low and high 64-bit chunks of an __m128i
static inline __m128i swap(const __m128i x) {
  return _mm_shuffle_epi32(x, _MM_SHUFFLE(1,0,3,2));
}

// Shift each 64-bit chunk left.  Use a macro to ensure n is an immediate.
template <int shift>
static inline __m128i left_shift(const __m128i x) {
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
GF1p128 operator*(const GF1p128 a, const GF1p128 b) {
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

// Compute a polynomial corresponding to jumping forward n steps.
// The result is represented in the fast GF(2^128).
static GF1p128 FastJumpPoly(const uint128 steps) {
  // We'll iterate through the set bits from lowest to highest, clearing each
  // one as we go.
  uint64 n0 = steps;
  uint64 n1 = steps >> 64;
  const auto next = [&n0, &n1]() {
    const int i0 = __builtin_ffsll(n0);
    const int i1 = __builtin_ffsll(n1);
    n0 ^= i0 ? static_cast<uint64>(1) << (i0 - 1) : 0;
    n1 ^= !i0 && i1 ? static_cast<uint64>(1) << (i1 - 1) : 0;
    return i0 ? i0 : i1 ? i1 + 64 : 0;
  };

  // read(i) corresponds to a jump by 2^relu(i-1) steps
  const auto read = [](int i) {
    if (i == 0) return GF1p128(1, 0);
    return GF1p128(*reinterpret_cast<const __m128i*>(kFastJumps + 2 * (i-1)));
  };

  // We unroll the loop by two.  Note that the compiler can't do this
  // transform: it doesn't know GF1p128 is a multiplicative abelian group.
  auto jump0 = read(next());
  auto jump1 = read(next());
  for (;;) {
    const auto i0 = next();
    if (!i0) break;
    const auto i1 = next();
    jump0 = jump0 * read(i0);
    jump1 = jump1 * read(i1);
  }
  return jump0 * jump1;
}

// Compute the matrix vector product Ab, where A in GF(2)^{128 * 128}.
// To take full advantage of AVX, A must be packed as
//
//   packed_a[4i:4i+4] = concat([A[:,i],A[:,i+64])
static __m128i PackedMatmul(const uint64 packed_a[256], const __m128i b) {
  // Each iteration of our loop processes two bits of b at a time; one from
  // the low 64 bits and one from the high.  To do this, we repeat each 64-bit
  // chunk of b two times.
  const auto b2 = _mm256_permute4x64_epi64(
      _mm256_castsi128_si256(b), _MM_SHUFFLE(1, 1, 0, 0));
  const auto ones = _mm256_set_epi64x(1, 1, 1, 1);

  // Perform matrix vector multiply with 256 bit instructions.  The low 128 bits
  // of the intermediate values correspond to the low 64 bits of b, and similarly
  // for the high bits.  We use a tree structured reduction to maximize
  // instruction level parallelism.
#define R0(i)                                                      \
    (_mm256_cmpeq_epi64(_mm256_srli_epi64(b2, (i)) & ones, ones) & \
     *reinterpret_cast<const __m256i*>(packed_a + 4 * (i)))
#define R1(i) (R0(i) ^ R0(i + 1))
#define R2(i) (R1(i) ^ R1(i + 2))
#define R3(i) (R2(i) ^ R2(i + 4))
#define R4(i) (R3(i) ^ R3(i + 8))
#define R5(i) (R4(i) ^ R4(i + 16))
#define R6(i) (R5(i) ^ R5(i + 32))
  const auto ab = R6(0);
#undef R6
#undef R5
#undef R4
#undef R3
#undef R2
#undef R1
#undef R0

  // Finish up by combining low and high 128 bit chunks.
  return _mm256_castsi256_si128(
      ab ^ _mm256_permute4x64_epi64(ab, _MM_SHUFFLE(1, 0, 3, 2)));
}

// Convert from the fast GF(2^128) to the slow GF(2^128).  The slow
// representation uses the characteristic polynomial of xorshift128+ as the
// modulus, and therefore corresponds to xorshift128+ jumps.
static __m128i FastToSlow(const GF1p128 fast) {
  return PackedMatmul(kFastToSlow, fast.c());
}

array<uint64, 2> XorshiftBaseJump(const uint128 steps) {
  return unpack(PackedMatmul(kFastToBaseJump, FastJumpPoly(steps).c()));
}

array<uint64, 2> XorshiftJump(array<uint64, 2> state, const uint128 steps) {
  // Compute the 128 step xorshift128+ sequence starting at state, packing
  // it in the order expected by PackedMatmul.  This is slower than it could
  // be since it writes memory.  For speed, use XorshiftBaseJump.
  uint64 sequence[256];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 64; j++) {
      sequence[4*j + 2*i + 0] = state[0];
      sequence[4*j + 2*i + 1] = state[1];
      state = XorshiftStep(state);
    }
  }

  // Perform the matmul
  return unpack(PackedMatmul(sequence, FastToSlow(FastJumpPoly(steps))));
}

}  // namespace tensorflow
