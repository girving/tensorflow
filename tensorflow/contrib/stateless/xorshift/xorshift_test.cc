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
// Tests for xorshift128+ random access

#include <vector>
#include "tensorflow/contrib/stateless/xorshift/xorshift.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using std::array;

TEST(Xorshift, Multiply) {
  // Small values are easy, since they don't touch the modulus
  const auto small = [](int n) { return GF1p128(n, 0); };
  const auto zero = small(0);
  const auto one = small(1);
  const auto two = small(2);
  for (int i = 0; i < 10; i++) {
    const auto s = small(i);
    const auto t = small(2 * i);
    EXPECT_EQ(zero * s, zero);
    EXPECT_EQ(s * zero, zero);
    EXPECT_EQ(s * one, s);
    EXPECT_EQ(one * s, s);
    EXPECT_EQ(s * two, t);
    EXPECT_EQ(two * s, t);
  }

  // Medium values are medium
  {
    const GF1p128 a(0, 1);
    const GF1p128 b(0b10000111, 0);
    EXPECT_EQ(a * a, b);
  }

  // Big values are harder
  {
    const GF1p128 a(0xc73f95cd13a24151, 0x00d168bc3d2bc05a);
    const GF1p128 b(0xb3307b76a009c5dd, 0x503c32d80cf222e3);
    const GF1p128 c(0x5194441cd2ed2a77, 0xf72f3cf0fd7f6408);
    EXPECT_EQ(a * b, c);
  }
}

void JumpTest(const array<uint64, 2> start, const int n) {
  std::vector<array<uint64, 2>> steps(n + 1);
  {
    // Compute a small path using step
    auto state = start;
    for (int i = 0; i <= n; i++) {
      steps[i] = state;
      state = XorshiftStep(state);
    }
  }

  // Check forward jumps
  for (int i = 0; i <= n; i++) {
    const auto jump = XorshiftJump(steps[0], i);
    ASSERT_EQ(steps[i], jump) << "i = " << i;
  }

  // Check backward jumps
  for (int i = 0; i <= n; i++) {
    const auto backwards = ~uint128(0) - i;
    const auto jump = XorshiftJump(steps[n], backwards);
    ASSERT_EQ(steps[n - i], jump) << "i = -" << i;
  }
}

TEST(Xorshift, Jump) {
  JumpTest({{1, 0}}, 10);
  JumpTest({{1, 0}}, 513);
  JumpTest({{0x0741701b7b9a5f6a, 0x2616008e7d2321f0}}, 77);
}

// Conventional stateful random number generator
class Random {
 public:
  Random(uint64 seed0, uint64 seed1) : state_({{seed0, seed1}}) {}

  uint64 random64() {
    const auto r = state_[0] + state_[1];
    state_ = XorshiftStep(state_);
    return r;
  }

  uint128 random128() {
    const auto lo = random64();
    const auto hi = random64();
    return lo | uint128(hi) << 64;
  }

 private:
  array<uint64, 2> state_;
};

TEST(Xorshift, LongJump) {
  // Check that jumps compose as expected.  Note that this
  // also implicitly checks commutativity and associativity.
  Random random(0x0741701b7b9a5f6a, 0x2616008e7d2321f0);
  for (int i = 0; i < 1000; i++) {
    const array<uint64, 2> s = {{random.random64(), random.random64()}};
    const auto n0 = random.random128();
    const auto n1 = random.random128();
    auto n = n0 + n1;
    n += n < n0;  // Correct for 2^128 vs. 2^128-1
    ASSERT_EQ(XorshiftJump(XorshiftJump(s, n0), n1), XorshiftJump(s, n));
  }
}

TEST(Xorshift, BaseJump) {
  // Check the first few cases
  {
    auto state = kXorshiftJumpBase;
    for (int i = 0; i < 10; i++) {
      ASSERT_EQ(XorshiftBaseJump(i), state);
      state = XorshiftStep(state);
    }
  }

  // Check that base jump matches jump.
  Random random(0x0741701b7b9a5f6a, 0x2616008e7d2321f0);
  for (int i = 0; i < 1000; i++) {
    const auto n = random.random128();
    ASSERT_EQ(XorshiftBaseJump(n), XorshiftJump(kXorshiftJumpBase, n));
  }
}

// Jump by random uint128 steps (with about half of the bits set)
void BM_Dense(const int iters) {
  Random random(0x0741701b7b9a5f6a, 0x2616008e7d2321f0);
  uint64 seen = 0;
  testing::StartTiming();
  for (uint64 i = 0; i < iters; i++) {
    const auto steps = random.random128();
    const auto state = XorshiftBaseJump(steps);
    seen |= state[0] | state[1];
  }
  testing::StopTiming();
  ASSERT_EQ(~seen, 0);  // Avoid dead code elimination
}
BENCHMARK(BM_Dense);

//  Jump by sparse uint128 steps (with up to 4 bits set)
void BM_Sparse(const int iters) {
  Random random(0x0741701b7b9a5f6a, 0x2616008e7d2321f0);
  uint64 seen = 0;
  testing::StartTiming();
  for (int iter = 0; iter < iters; iter++) {
    const auto bits = random.random64();
    const int i0 = bits & 127;
    const int i1 = bits >> 8 & 127;
    const int i2 = bits >> 16 & 127;
    const int i3 = bits >> 32 & 127;
    const uint128 one = 1;
    const auto steps = one << i0 | one << i1 | one << i2 | one << i3;
    const auto state = XorshiftBaseJump(steps);
    seen |= state[0] | state[1];
  }
  testing::StopTiming();
  ASSERT_EQ(~seen, 0);  // Avoid dead code elimination
}
BENCHMARK(BM_Sparse);

}  // namespace
}  // namespace tensorflow
