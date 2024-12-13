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

use super::{Fr, MODULUS, R2};
use crate::util::sbb;

#[cfg(feature = "zeroize")]
impl zeroize::DefaultIsZeroes for Fr {}

impl Fr {
    /// Creates a `Fr` from arbitrary bytes by hashing the input with BLAKE2b
    /// into a 512-bits number, and then converting the number into its scalar
    /// representation by reducing it by the modulo.
    ///
    /// By treating the output of the BLAKE2b hash as a random oracle, this
    /// implementation follows the first conversion of
    /// https://hackmd.io/zV6qe1_oSU-kYU6Tt7pO7Q with concrete numbers:
    /// ```text
    /// p = 0x0e7db4ea6533afa906673b0101343b00a6682093ccc81082d0970e5ed6f72cb7
    /// p = 6554484396890773809930967563523245729705921265872317281365359162392183254199
    ///
    /// l = 2
    ///
    /// s^l = (2^256)^2 = 2^512
    /// s = 13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084096
    ///
    /// r' = 2045593080716281616348203381729468609728209645786990242449482205581148743408809
    ///
    /// m' = 2244478849891746936202736009816130624903096691796347063256129649283183245105
    /// ```
    pub fn hash_to_scalar(input: &[u8]) -> Self {
        let state = blake2b_simd::Params::new()
            .hash_length(64)
            .to_state()
            .update(input)
            .finalize();

        let bytes = state.as_bytes();

        Self::from_u512([
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[32..40]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[40..48]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[48..56]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[56..64]).unwrap()),
        ])
    }

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

#[cfg(feature = "serde")]
mod serde_support {
    extern crate alloc;

    use alloc::string::{String, ToString};

    use dusk_bytes::Serializable;
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::Fr;

    impl Serialize for Fr {
        fn serialize<S: Serializer>(
            &self,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            let s = hex::encode(self.to_bytes());
            serializer.serialize_str(&s)
        }
    }

    impl<'de> Deserialize<'de> for Fr {
        fn deserialize<D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Self, D::Error> {
            let s = String::deserialize(deserializer)?;
            let decoded = hex::decode(s).map_err(Error::custom)?;
            let decoded_len = decoded.len();
            let bytes: [u8; Self::SIZE] = decoded.try_into().map_err(|_| {
                Error::invalid_length(
                    decoded_len,
                    &Self::SIZE.to_string().as_str(),
                )
            })?;
            Fr::from_bytes(&bytes)
                .into_option()
                .ok_or(Error::custom("Failed to deserialize Fr: invalid Fr"))
        }
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

#[cfg(all(test, feature = "alloc"))]
mod fuzz {
    use alloc::vec::Vec;

    use crate::fr::{Fr, MODULUS};
    use crate::util::sbb;

    fn is_scalar_in_range(scalar: &Fr) -> bool {
        // subtraction against modulus must underflow
        let borrow = scalar
            .0
            .iter()
            .zip(MODULUS.0.iter())
            .fold(0, |borrow, (&s, &m)| sbb(s, m, borrow).1);

        borrow == u64::MAX
    }

    quickcheck::quickcheck! {
        fn prop_hash_to_scalar(bytes: Vec<u8>) -> bool {
            let scalar = Fr::hash_to_scalar(&bytes);

            is_scalar_in_range(&scalar)
        }
    }
}

#[cfg(feature = "zeroize")]
#[test]
fn test_zeroize() {
    use zeroize::Zeroize;

    let mut scalar = Fr::one();
    scalar.zeroize();
    assert_eq!(scalar, Fr::zero());
}

#[cfg(feature = "serde")]
#[test]
fn serde_fr() {
    use ff::Field;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(0xdead);
    let fr = Fr::random(&mut rng);
    let ser = serde_json::to_string(&fr).unwrap();
    let deser = serde_json::from_str(&ser).unwrap();
    assert_eq!(fr, deser);

    // Should error when the encoding is wrong
    let wrong_encoded = "wrong-encoded";
    let fr: Result<Fr, _> = serde_json::from_str(&wrong_encoded);
    assert!(fr.is_err());

    // Should error when the input is too long
    let length_33_enc = "\"e4ab9de40283a85d6ea0cd0120500697d8b01c71b7b4b520292252d20937000631\"";
    let fr: Result<Fr, _> = serde_json::from_str(&length_33_enc);
    assert!(fr.is_err());

    // Should error when the input is too short
    let length_31_enc =
        "\"1751c37a1dca7aa4c048fcc6177194243edc3637bae042e167e4285945e046\"";
    let fr: Result<Fr, _> = serde_json::from_str(&length_31_enc);
    assert!(fr.is_err());
}
