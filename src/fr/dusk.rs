// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use core::cmp::{Ord, Ordering, PartialOrd};
use core::convert::TryInto;
use core::ops::{Index, IndexMut};

use dusk_bls12_381::BlsScalar;
use dusk_bytes::{Error as BytesError, Serializable};
use rand_core::{CryptoRng, RngCore, SeedableRng};

use super::{Fr, MODULUS, R2};
use crate::util::sbb;

/// Random number generator for generating scalars that are uniformly
/// distributed over the entire field of scalars.
///
/// Because scalars take 251 bits for encoding it is difficult to generate
/// random bit-pattern that ensures to encode a valid scalar.
/// Wrapping the values that are higher than [`MODULUS`], as done in
/// [`Self::random`], results in hitting some values more than others, whereas
/// zeroing out the highest two bits will eliminate some values from the
/// possible results.
///
/// This function achieves a uniform distribution of scalars by using rejection
/// sampling: random bit-patterns are generated until a valid scalar is found.
/// The scalar creation is not constant time but that shouldn't be a concern
/// since no information about the scalar can be gained by knowing the time of
/// its generation.
///
/// ## Example
///
/// ```
/// use rand::rngs::{StdRng, OsRng};
/// use rand::SeedableRng;
/// use dusk_jubjub::{JubJubScalar, UniScalarRng};
/// use ff::Field;
///
/// // using a seedable random number generator
/// let mut rng: UniScalarRng<StdRng> = UniScalarRng::seed_from_u64(0x42);
/// let _scalar = JubJubScalar::random(rng);
///
/// // using randomness derived from the os
/// let mut rng = UniScalarRng::<OsRng>::default();
/// let _ = JubJubScalar::random(rng);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct UniScalarRng<R>(R);

impl<R> CryptoRng for UniScalarRng<R> where R: CryptoRng {}

impl<R> RngCore for UniScalarRng<R>
where
    R: RngCore,
{
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    // We use rejection sampling to generate a valid scalar.
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        if dest.len() < 32 {
            panic!("buffer too small to generate uniformly distributed random scalar");
        }

        // Loop until we find a canonical scalar.
        // As long as the random number generator is implemented properly, this
        // loop will terminate.
        let mut scalar = [0u64; 4];
        loop {
            for integer in scalar.iter_mut() {
                *integer = self.0.next_u64();
            }

            // Check that the generated potential scalar is smaller than MODULUS
            let bx = scalar[3] <= MODULUS.0[3];
            let b1 = bx && MODULUS.0[0] > scalar[0];
            let b2 = bx && (MODULUS.0[1] + b1 as u64) > scalar[1];
            let b3 = bx && (MODULUS.0[2] + b2 as u64) > scalar[2];
            let b4 = bx && (MODULUS.0[3] + b3 as u64) > scalar[3];

            if b4 {
                // Copy the generated random scalar in the first 32 bytes of the
                // destination slice (scalars are stored in little endian).
                for (i, integer) in scalar.iter().enumerate() {
                    let bytes = integer.to_le_bytes();
                    dest[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
                }

                // Zero the remaining bytes (if any).
                if dest.len() > 32 {
                    dest[32..].fill(0);
                }
                return;
            }
        }
    }

    fn try_fill_bytes(
        &mut self,
        dest: &mut [u8],
    ) -> Result<(), rand_core::Error> {
        self.0.try_fill_bytes(dest)
    }
}

impl<R> SeedableRng for UniScalarRng<R>
where
    R: SeedableRng,
{
    type Seed = <R as SeedableRng>::Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        Self(R::from_seed(seed))
    }
}

impl Fr {
    /// SHR impl: shifts bits n times, equivalent to division by 2^n.
    #[inline]
    pub fn divn(&mut self, mut n: u32) {
        if n >= 256 {
            *self = Self::from(0u64);
            return;
        }

        while n >= 64 {
            let mut t = 0;
            for i in self.0.iter_mut().rev() {
                core::mem::swap(&mut t, i);
            }
            n -= 64;
        }

        if n > 0 {
            let mut t = 0;
            for i in self.0.iter_mut().rev() {
                let t2 = *i << (64 - n);
                *i >>= n;
                *i |= t;
                t = t2;
            }
        }
    }

    /// Reduces bit representation of numbers, such that
    /// they can be evaluated in terms of the least significant bit.
    pub fn reduce(&self) -> Self {
        Fr::montgomery_reduce(
            self.0[0], self.0[1], self.0[2], self.0[3], 0u64, 0u64, 0u64, 0u64,
        )
    }

    /// Evaluate if a `Scalar, from Fr` is even or not.
    pub fn is_even(&self) -> bool {
        self.0[0] % 2 == 0
    }

    /// Compute the result from `Scalar (mod 2^k)`.
    ///
    /// # Panics
    ///
    /// If the given k is > 32 (5 bits) as the value gets
    /// greater than the limb.  
    pub fn mod_2_pow_k(&self, k: u8) -> u8 {
        (self.0[0] & ((1 << k) - 1)) as u8
    }

    /// Compute the result from `Scalar (mods k)`.
    ///
    /// # Panics
    ///
    /// If the given `k > 32 (5 bits)` || `k == 0` as the value gets
    /// greater than the limb.   
    pub fn mods_2_pow_k(&self, w: u8) -> i8 {
        assert!(w < 32u8);
        let modulus = self.mod_2_pow_k(w) as i8;
        let two_pow_w_minus_one = 1i8 << (w - 1);

        match modulus >= two_pow_w_minus_one {
            false => modulus,
            true => modulus - ((1u8 << w) as i8),
        }
    }

    /// Computes the windowed-non-adjacent form for a given an element in
    /// the JubJub Scalar field.
    ///
    /// The wnaf of a scalar is its breakdown:
    ///     scalar = sum_i{wnaf[i]*2^i}
    /// where for all i:
    ///     -2^{w-1} < wnaf[i] < 2^{w-1}
    /// and
    ///     wnaf[i] * wnaf[i+1] = 0
    pub fn compute_windowed_naf(&self, width: u8) -> [i8; 256] {
        let mut k = self.reduce();
        let mut i = 0;
        let one = Fr::one().reduce();
        let mut res = [0i8; 256];

        while k >= one {
            if !k.is_even() {
                let ki = k.mods_2_pow_k(width);
                res[i] = ki;
                k -= Fr::from(ki);
            } else {
                res[i] = 0i8;
            };

            k.divn(1u32);
            i += 1;
        }
        res
    }
}

// TODO implement From<T> for any integer type smaller than 128-bit
impl From<i8> for Fr {
    // FIXME this could really be better if we removed the match
    fn from(val: i8) -> Fr {
        match (val >= 0, val < 0) {
            (true, false) => Fr([val.unsigned_abs() as u64, 0u64, 0u64, 0u64]),
            (false, true) => -Fr([val.unsigned_abs() as u64, 0u64, 0u64, 0u64]),
            (_, _) => unreachable!(),
        }
    }
}

impl From<Fr> for BlsScalar {
    fn from(scalar: Fr) -> BlsScalar {
        let bls_scalar =
            <BlsScalar as Serializable<32>>::from_bytes(&scalar.to_bytes());

        // The order of a JubJub's Scalar field is shorter than a BLS
        // Scalar, so convert any jubjub scalar to a BLS' Scalar
        // should always be safe.
        assert!(
            bls_scalar.is_ok(),
            "Failed to convert a Scalar from JubJub to BLS"
        );

        bls_scalar.unwrap()
    }
}

impl Index<usize> for Fr {
    type Output = u64;
    fn index(&self, _index: usize) -> &u64 {
        &(self.0[_index])
    }
}

impl IndexMut<usize> for Fr {
    fn index_mut(&mut self, _index: usize) -> &mut u64 {
        &mut (self.0[_index])
    }
}

impl PartialOrd for Fr {
    fn partial_cmp(&self, other: &Fr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fr {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self;
        for i in (0..4).rev() {
            #[allow(clippy::comparison_chain)]
            if a[i] > other[i] {
                return Ordering::Greater;
            } else if a[i] < other[i] {
                return Ordering::Less;
            }
        }
        Ordering::Equal
    }
}

impl Serializable<32> for Fr {
    type Error = BytesError;

    /// Attempts to convert a little-endian byte representation of
    /// a field element into an element of `Fr`, failing if the input
    /// is not canonical (is not smaller than r).
    fn from_bytes(bytes: &[u8; Self::SIZE]) -> Result<Self, Self::Error> {
        let mut tmp = Fr([0, 0, 0, 0]);

        tmp.0[0] = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        tmp.0[1] = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        tmp.0[2] = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        tmp.0[3] = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        if is_some == 0 {
            return Err(BytesError::InvalidData);
        }

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        tmp *= &R2;

        Ok(tmp)
    }

    /// Converts an element of `Fr` into a byte representation in
    /// little-endian byte order.
    fn to_bytes(&self) -> [u8; Self::SIZE] {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = Fr::montgomery_reduce(
            self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0,
        );

        let mut res = [0; Self::SIZE];
        res[0..8].copy_from_slice(&tmp.0[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp.0[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp.0[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp.0[3].to_le_bytes());

        res
    }
}

#[test]
fn w_naf_3() {
    let scalar = Fr::from(1122334455u64);
    let w = 3;
    // -1 - 1*2^3 - 1*2^8 - 1*2^11 + 3*2^15 + 1*2^18 - 1*2^21 + 3*2^24 +
    // 1*2^30
    let expected_result = [
        -1i8, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 3, 0, 0, 1, 0, 0,
        -1, 0, 0, 3, 0, 0, 0, 0, 0, 1,
    ];

    let mut expected = [0i8; 256];
    expected[..expected_result.len()].copy_from_slice(&expected_result);

    let computed = scalar.compute_windowed_naf(w);

    assert_eq!(expected, computed);
}

#[test]
fn w_naf_4() {
    let scalar = Fr::from(58235u64);
    let w = 4;
    // -5 + 7*2^7 + 7*2^13
    let expected_result = [-5, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7];

    let mut expected = [0i8; 256];
    expected[..expected_result.len()].copy_from_slice(&expected_result);

    let computed = scalar.compute_windowed_naf(w);

    assert_eq!(expected, computed);
}

#[test]
fn w_naf_2() {
    let scalar = -Fr::one();
    let w = 2;
    let two = Fr::from(2u64);

    let wnaf = scalar.compute_windowed_naf(w);

    let recomputed = wnaf.iter().enumerate().fold(Fr::zero(), |acc, (i, x)| {
        if *x > 0 {
            acc + Fr::from(*x as u64) * two.pow(&[(i as u64), 0u64, 0u64, 0u64])
        } else if *x < 0 {
            acc - Fr::from(-(*x) as u64)
                * two.pow(&[(i as u64), 0u64, 0u64, 0u64])
        } else {
            acc
        }
    });
    assert_eq!(scalar, recomputed);
}

#[test]
fn test_uni_rng() {
    use rand::rngs::StdRng;
    let mut rng: UniScalarRng<StdRng> = UniScalarRng::seed_from_u64(0xbeef);

    let mut buf32 = [0u8; 32];
    let mut buf64 = [0u8; 64];

    for _ in 0..100000 {
        // fill an array of 64 bytes with our random scalar generator
        rng.fill_bytes(&mut buf64);

        // copy the first 32 bytes into another buffer and check that these
        // bytes are the canonical encoding of a scalar
        buf32.copy_from_slice(&buf64[..32]);
        let scalar1: Option<Fr> = Fr::from_bytes(&buf32).into();
        assert!(scalar1.is_some());

        // create a second scalar from the 64 bytes wide array and check that it
        // generates the same scalar as generated from the 32 bytes wide
        // array
        let scalar2: Fr = Fr::from_bytes_wide(&buf64);
        let scalar1 = scalar1.unwrap();
        assert_eq!(scalar1, scalar2);
    }
}
