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
// Generate the lookup tables used by xorshift.cc.

#include "tensorflow/contrib/stateless/xorshift/xorshift.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/LU"

namespace tensorflow {
namespace {

using std::array;
template <class T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <class T> using Matrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// Small xorshift for testing purposes
array<uint32, 1> SmallXorshiftStep(const array<uint32, 1> state) {
  auto x = state[0];
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return {{x}};
}

class GF2 {
 public:
  GF2() : x_(0) {}
	explicit GF2(bool x) : x_(x) {}
	GF2 operator+(GF2 a) const { return GF2(x_ ^ a.x_); }
	GF2 operator-(GF2 a) const { return *this + a; }
	GF2 operator*(GF2 a) const { return GF2(x_ & a.x_); }
	GF2& operator+=(GF2 a) { return *this = *this + a; }
	GF2& operator-=(GF2 a) { return *this = *this - a; }
	GF2& operator*=(GF2 a) { return *this = *this * a; }
  GF2 operator/(GF2 a) const { CHECK(a.x_); return *this; }
  GF2& operator/=(GF2 a) { CHECK(a.x_); return *this; }
  bool operator==(GF2 a) const { return x_ == a.x_; }
  bool operator!=(GF2 a) const { return x_ != a.x_; }
  explicit operator bool() const { return x_; }

  // For pivoted LU purposes
  friend GF2 abs(GF2 a) { return a; }
  bool operator<(GF2 a) const { return x_ < a.x_; }
  bool operator>(GF2 a) const { return x_ > a.x_; }
  GF2 operator*(long) const {
    LOG(FATAL) << "Not needed, since we set an explicit threshold";
  }

 private:
	bool x_;
};

// Polynomials over GF2 with degree <= d
template <int degree> class Poly {
 public:
  // Constant polynomials
  Poly() = default;
  explicit Poly(GF2 c) { coeffs_.set(0, static_cast<bool>(c)); }
  explicit Poly(int c) : Poly(GF2(c & 1)) {}

  // Explicit coefficients
  explicit Poly(const std::initializer_list<bool>& coeffs) {
    int i = 0;
    for (const bool c : coeffs) {
      set_coeff(i++, GF2(c));
    }
  }

  GF2 coeff(int n) const {
    CHECK(0 <= n);
    return GF2(n <= degree ? coeffs_[n] : 0);
  }

  void set_coeff(int n, GF2 x) {
    CHECK(0 <= n && n <= degree);
    coeffs_.set(n, static_cast<bool>(x));
  }

  // A power of x
  static Poly pow_x(int n) {
    CHECK(0 <= n && n <= degree);
    Poly p;
    p.coeffs_.set(n);
    return p;
  }

  bool operator==(const Poly& p) const { return coeffs_ == p.coeffs_; }

  Poly operator+(const Poly& p) const {
    Poly sum = *this;
    sum.coeffs_ ^= p.coeffs_;
    return sum;
  }

  Poly& operator+=(const Poly& p) {
    coeffs_ ^= p.coeffs_;
    return *this;
  }

  Poly operator-(const Poly& p) const { return *this + p; }
  Poly& operator-=(const Poly& p) { return *this += p; }

  Poly operator<<(int n) const {
    Poly shift;
    shift.coeffs_ = coeffs_ << n;
    return shift;
  }

  Poly operator>>(int n) const {
    Poly shift;
    shift.coeffs_ = coeffs_ >> n;
    return shift;
  }

  Poly operator*(const Poly& p) const {
    Poly prod;
    for (int i = 0; i <= degree; i++) {
      if (coeff(i)) {
        prod += p << i;
      }
    }
    return prod;
  }

  // The modulus has an implicit high x^degree component
  Poly multiply_mod(const Poly& p, const Poly& mod) const {
    Poly lo, hi;
    for (int i = 0; i <= degree; i++) {
      if (coeff(i)) {
        lo += p << i;
        hi += p >> (degree + 1 - i);
      }
    }
    for (int i = degree - 1; i >= 0; i--) {
      if (hi.coeff(i)) {
        lo += mod << i;
        hi += mod >> (degree + 1 - i);
      }
    }
    return lo;
  }

  // Apply a polynomial to a matrix
  Matrix<GF2> operator()(const Matrix<GF2>& m) const {
    const int n = m.rows();
    CHECK_EQ(n, m.cols());
    Matrix<GF2> sum(n, n), power(n, n);
    sum = Matrix<GF2>::Zero(n, n);
    power = Matrix<GF2>::Identity(n, n); 
    for (int i = 0; i <= degree; i++) {
      if (coeff(i)) sum += power;
      power *= m;
    }
    return sum;
  }

 private:
  std::bitset<degree + 1> coeffs_;
};

// Elements of an extension field over GF(2)
template <int bits>
class Extension {
 public:
  Extension(const Poly<bits-1> poly, const Poly<bits-1> modulus)
      : poly_(poly), modulus_(modulus) {}

  const Poly<bits-1>& poly() const { return poly_; }

  bool operator==(const Extension& e) const {
    return poly() == e.poly() && modulus_ == e.modulus_;
  }

  Extension operator+(const Extension& e) const {
    CHECK(modulus_ == e.modulus_);
    return Extension(poly() + e.poly(), modulus_);
  }

  Extension operator*(const Extension& e) const {
    CHECK(modulus_ == e.modulus_);
    return Extension(poly().multiply_mod(e.poly(), modulus_), modulus_);
  }

 private:
  Poly<bits-1> poly_;
  Poly<bits-1> modulus_;
};

// xorshift128+ as a 128 by 128 matrix over GF(2)
template <class Word, size_t words> Matrix<GF2>
StepMatrix(array<Word, words> (*step)(array<Word, words>)) {
  const int word_bits = 8 * sizeof(Word);
  const int bits = words * word_bits;
  Matrix<GF2> matrix(bits, bits);
  for (int j = 0; j < bits; j++) {
    array<Word, words> single;
    single.fill(0);
    single[j / word_bits] = static_cast<Word>(1) << (j % word_bits);
    const auto column = step(single);
    for (int i = 0; i < bits; i++) {
      matrix(i, j) = GF2(column[i / word_bits] >> (i % word_bits) & 1);
    }
  }
  return matrix;
}

// Conversion from xorshift state to vectors
template <class Word, size_t words>
Vector<GF2> ToVec(const array<Word, words> state) {
  const int word_bits = 8 * sizeof(Word);
  const int bits = words * word_bits;
  Vector<GF2> vec(bits);
  for (int i = 0; i < bits; i++) {
    vec(i) = GF2(state[i / word_bits] >> (i % word_bits) & 1);
  }
  return vec;
}

// We use the division free determinant routine from
//   Bird 2011, "A simple division-free algorithm for computing determinants".
template <class T>
T Determinant(const Matrix<T>& a) {
  CHECK_EQ(a.rows(), a.cols());
  const int n = a.rows();

  // Iterate F_A(X) = mu(X) * A from Bird 2011
  auto x = a;
  Matrix<T> mu(n, n);
  for (int k = 0; k < n - 1; k++) {
    mu = x.template triangularView<Eigen::StrictlyUpper>();
    for (int i = n - 1; i > 0; i--) {
      mu(i - 1, i - 1) = mu(i, i) - x(i, i);
    }
    x = mu * a;
  }

  // The determinant is the first entry (the rest are zeros)
  return x(0, 0);
}

template <int bits>
Poly<bits> CharacteristicPoly(const Matrix<GF2>& m) {
  // Construct the matrix m + x
  const int n = m.rows();
  Matrix<Poly<bits>> mx(n, n);
  mx = m.cast<Poly<bits>>();
  const auto x = Poly<bits>::pow_x(1);
  for (int i = 0; i < n; i++) {
    mx(i, i) += x;
  }
  return Determinant(mx);
}

// Conversion from polynomials to vectors
template <int bits>
Vector<GF2> ToVec(const Poly<bits>& poly) {
  Vector<GF2> vec(bits + 1);
  for (int i = 0; i <= bits; i++) {
    vec(i) = poly.coeff(i);
  }
  return vec;
}

// Conversion from vectors to polynomials
template <int bits>
Poly<bits> ToPoly(const Vector<GF2>& vec, bool truncate = false) {
  if (!truncate) {
    CHECK_EQ(vec.size(), bits + 1);
  }
  Poly<bits> poly;
  for (int i = 0; i <= bits; i++) {
    if (vec(i)) {
      poly += Poly<bits>::pow_x(i);
    }
  }
  return poly;
}

// Compute an explicit isomorphism from GF2[x]/poly0 to GF2[x]/poly1.
// We use Algorithm 4.1 of
//   Allombert 2002, Explicit computation of isomorphisms between finite fields.
// We use the same variable names when convenient.
template <int d>
Matrix<GF2> Isomorphism(const Poly<d> poly0, const Poly<d> poly1) {
  const int bits = d + 1;
  const int power = static_cast<int>(std::log2(static_cast<double>(bits)));
  CHECK_EQ(bits, 1 << power);

  // Solve the additive Hilbert equation x^2 + x = a
  // Compute a Frobenius isomorphism as a matrix over the power basis,
  // then precompute 1 / (F - 1).  This matrix is singular since F - 1 is
  // the identity on both 0 and 1, so we use rank revealing LU.
  //
  // There is a nice explanation of what's going on here at
  //   https://math.stackexchange.com/questions/639310/
  //      applications-of-additive-version-of-hilberts-theorem-90
  const auto hilbert_lu = [bits](const Poly<bits-1> poly) {
    Matrix<GF2> frob(bits, bits);
    for (int j = 0; j < bits; j++) {
      auto a = Extension<bits>(Poly<bits-1>::pow_x(j), poly);
      auto sqr_a = a * a;
      for (int i = 0; i < bits; i++) {
        frob(i, j) = sqr_a.poly().coeff(i);
      }
    }
    for (int i = 0; i < bits; i++) {
      frob(i, i) += GF2(1);
    }
    Eigen::FullPivLU<Matrix<GF2>> lu;
    lu.setThreshold(GF2());  // Anything nonzero is huge
    lu.compute(frob);
    CHECK_EQ(lu.rank(), bits - 1) << "Polynomial is reducible";
    return lu;
  };

  // Construct a map from the alpha basis to the power basis
  const auto alpha_basis = [hilbert_lu, bits, power](const Poly<bits-1>& poly) {
    // Construct alpha by iterating through larger and larger subfields
    const auto hilbert = hilbert_lu(poly);
    auto a = Extension<bits>(Poly<bits-1>(1), poly);
    auto alpha = a;
    for (int e = 0; e < power; e++) {
      a = a * alpha;
      alpha = Extension<bits>(
          ToPoly<bits-1>(hilbert.solve(ToVec(a.poly()))), poly);
    }

    // Build matrix from alpha to power basis
    Matrix<GF2> to_alpha(bits, bits);
    auto alpha_power = Extension<bits>(Poly<bits-1>(1), poly);
    for (int j = 0; j < bits; j++) {
      to_alpha.col(j) = ToVec(alpha_power.poly());
      alpha_power = alpha_power * alpha;
    }
    return to_alpha;
  };

  // Map from one power basis to the other via the alpha basis
  const auto alpha_to_power0 = alpha_basis(poly0);
  const auto alpha_to_power1 = alpha_basis(poly1);
  return alpha_to_power1 * alpha_to_power0.inverse();
}

// Test that XorshiftStep and StepMatrix match
template <class Word, size_t words> void
TestStepMatrix(array<Word, words> (*step)(array<Word, words>),
               const array<Word, words> base) {
  const int bits = words * sizeof(Word) * 8;
  const auto matrix = StepMatrix(step);
  auto state = base;
  for (int i = 0; i < 2 * bits; i++) {
    auto next = step(state);
    ASSERT_EQ(ToVec(next), matrix * ToVec(state));
    state = next;
  }
}

TEST(GenXorshift, StepMatrix) {
  TestStepMatrix(XorshiftStep, kXorshiftJumpBase);
}

TEST(GenXorshift, SmallStepMatrix) {
  TestStepMatrix(SmallXorshiftStep, {{0xc9ab72b7}});
}

TEST(GenXorshift, MultiplyMod) {
  auto state = kXorshiftJumpBase;
  for (int i = 0; i < 100; i++) {
    // Pick three random degree 10 polynomials
    auto bits = state[0] + state[1];
    state = XorshiftStep(state);
    Poly<10> x, y, mod;
    for (int i = 0; i < 10; i++) {
      x.set_coeff(i, GF2(bits >> i));
      y.set_coeff(i, GF2(bits >> (i + 10)));
      mod.set_coeff(i, GF2(bits >> (i + 20)));
    }

    // Test multiplication against simpler, slower implementation
    Poly<20> x2, y2, mod2;
    for (int i = 0; i <= 10; i++) {
      x2.set_coeff(i, x.coeff(i));
      y2.set_coeff(i, y.coeff(i));
      mod2.set_coeff(i, mod.coeff(i));
    }
    mod2.set_coeff(11, GF2(1));
    auto xy = x.multiply_mod(y, mod);
    auto xy2 = x2 * y2;
    for (int i = 20; i > 10; i--) {
      if (xy2.coeff(i)) {
        xy2 -= mod2 << (i - 11);
      }
    }
    for (int i = 0; i <= 20; i++) {
      EXPECT_TRUE(xy.coeff(i) == xy2.coeff(i));
    }
  }
}

TEST(GenXorshift, CharPolySimple) {
  Matrix<GF2> eye(2, 2);
  eye = Matrix<GF2>::Identity(2, 2);
  auto poly = CharacteristicPoly<5>(eye);
  EXPECT_EQ(poly, Poly<5>({1, 0, 1}));
}

TEST(GenXorshift, CharPolySmall) {
  auto matrix = StepMatrix(SmallXorshiftStep);
  auto poly = CharacteristicPoly<32>(matrix);

  // Compare with known stable value (leaving out the implicit x^32)
  EXPECT_EQ(poly, Poly<32>({
      1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
      0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
  }));

  // Check the Cayley-Hamilton theorem
  auto cayley = poly(matrix);
  ASSERT_EQ(cayley.rows(), 32);
  ASSERT_EQ(cayley.cols(), 32);
  EXPECT_TRUE((cayley.array() == GF2()).all());
}

template <int d>
void TestIsomorphism(const Poly<d> poly0, const Poly<d> poly1) {
  const auto forward = Isomorphism(poly0, poly1);

  // Is it a homomorphism?
  const auto f = [&](const Extension<d+1>& x) {
    return Extension<d+1>(ToPoly<d>(forward * ToVec(x.poly())), poly1);
  };
  array<uint32, 1> state = {{0x7976749a}};
  for (int i = 0; i < 100; i++) {
    const auto next = SmallXorshiftStep(state);
    const auto x = Extension<d+1>(ToPoly<d>(ToVec(state), true), poly0);
    const auto y = Extension<d+1>(ToPoly<d>(ToVec(next), true), poly0);
    CHECK(f(x + y) == f(x) + f(y));
    CHECK(f(x * y) == f(x) * f(y));
    state = next;
  }

  // Is it a bijection?
  auto backward = forward.inverse();
  CHECK(backward * forward == Matrix<GF2>::Identity(d+1, d+1));
}

TEST(GenXorshift, IsomorphismTrivial) {
  auto linear = Poly<0>({0});  // + x implicitly
  TestIsomorphism(linear, linear);
  auto quadratic = Poly<1>({1, 1});  // + x^2 implicitly
  TestIsomorphism(quadratic, quadratic);
}

TEST(GenXorshift, IsomorphismTiny) {
  auto lo = Poly<3>({1, 1, 0, 0});  // + x^4 implicitly
  auto hi = Poly<3>({1, 0, 0, 1});  // + x^4 implicitly
  TestIsomorphism(lo, hi);
}

TEST(GenXorshift, IsomorphismSmall) {
  // Construct a map from a pentanomial field to a more useful one
  auto matrix = StepMatrix(SmallXorshiftStep);
  auto poly = CharacteristicPoly<31>(matrix);
  auto simple = Poly<31>({1, 0, 1, 1, 0, 0, 0, 1});  // + x^32 implicitly
  TestIsomorphism(simple, poly);
}

void GenerateHeader(const string& path) {
  // Compute an isomorphism from the pentanomial field to the xorshift128+ field
  const auto fast = Poly<127>({1, 1, 1, 0, 0, 0, 0, 1});  // + x^128 implicitly
  const auto slow = CharacteristicPoly<127>(StepMatrix(XorshiftStep));
  const auto fast_to_slow = Isomorphism(fast, slow);
  const auto slow_to_fast = Matrix<GF2>(fast_to_slow.inverse());

  // All power of two jumps in the fast GF(2^128)
  Matrix<GF2> slow_jumps(128, 128);
  Extension<128> jump(Poly<127>::pow_x(1), slow);
  for (int n = 0; n < 128; n++) {
    slow_jumps.col(n) = ToVec(jump.poly());
    jump = jump * jump;
  }

  // The first 128 steps after kXorshiftJumpBase
  Matrix<GF2> slow_to_base_jump(128, 128);
  auto state = kXorshiftJumpBase;
  for (int n = 0; n < 128; n++) {
    slow_to_base_jump.col(n) = ToVec(state);
    state = XorshiftStep(state); 
  }

  // Convert a matrix to a string in hex form
  const auto show_matrix = [](const string& name, const Matrix<GF2> mat,
                              const bool pack) {
    CHECK_EQ(mat.rows(), 128);
    CHECK_EQ(mat.cols(), 128);
    auto str = strings::StrCat("static const uint64 ", name, "[256] = {\n");
    for (int n = 0; n < 128; n++) {
      const int j = pack ? n / 2 + n % 2 * 64 : n;
      uint64 column[2] = {0, 0};
      for (int i = 0; i < 128; i++) {
        const bool entry = static_cast<bool>(mat(i, j));
        column[i >> 6] |= static_cast<uint64>(entry) << (i & 63);
      }
      strings::StrAppend(&str, "    0x",
                         strings::Hex(column[0], strings::ZERO_PAD_16),
                         ", 0x", strings::Hex(column[1], strings::ZERO_PAD_16),
                         ",\n");
    }
    strings::StrAppend(&str, "};\n\n");
    return str;
  };

  // Boilerplate strings
  const char* guard =
      "TENSORFLOW_CONTRIB_STATELESS_XORSHIFT_XORSHIFT_GENERATED_H_";
  const char* license = R"mark(
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
)mark";

  const auto header = strings::StrCat(
    license + 1, 
    "// Warning: Autogenerated by gen_xorshift; DO NOT EDIT!\n",
    "// To regenerate, run\n",
    "//   blaze run -c opt gen_xorshift -- `realpath xorshift_generated.h`\n\n",
    "#ifndef ", guard, "\n#define ", guard, "\n\n",
    "#include \"tensorflow/core/platform/types.h\"\n\n",
    "namespace tensorflow {\nnamespace {\n\n",
    "// xorshift128+ jumps for each power of 2, represented as\n",
    "// GF(2^128, modulus=x^128 + x^7 + x^2 + x + 1).\n",
    show_matrix("kFastJumps", slow_to_fast * slow_jumps, false),
    "// Linear map from fast GF(2^128) to slow GF(2^128) with\n",
    "// characteristic polynomial modulus corresponding to xorshift128+.\n",
    show_matrix("kFastToSlow", fast_to_slow, true),
    "// Linear map from fast GF(2^128) to step^n(jump_base)\n",
    show_matrix("kFastToBaseJump", slow_to_base_jump * fast_to_slow, true),
    "}  // namespace\n}  // namespace tensorflow\n\n",
    "#endif  // ", guard, "\n");
  TF_CHECK_OK(WriteStringToFile(Env::Default(), path, header));
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  if (argc == 1) {
    // Run tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS(); 
  } else if (argc == 2) {
    // Generate a header file of precomputed tables
    tensorflow::GenerateHeader(argv[1]);
  } else {
    LOG(FATAL) << "Usage: " << argv[0] << " <xorshift_generated.h>";
  }
  return 0;
}
