// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use dusk_bytes::{Error, Serializable};
use dusk_jubjub::{
    AffinePoint, ExtendedPoint, Fq, Fr, SubgroupPoint, GENERATOR_EXTENDED,
};
use group::{Group, GroupEncoding};
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;

#[test]
fn scalar_serializable_matches_canonical_encoding() {
    for scalar in [Fr::zero(), Fr::one(), -Fr::one(), Fr::from(u64::MAX)] {
        let bytes = <Fr as Serializable<32>>::to_bytes(&scalar);
        assert_eq!(bytes, scalar.to_bytes());
        assert_eq!(<Fr as Serializable<32>>::from_bytes(&bytes), Ok(scalar));
    }
    let mut modulus = (-Fr::one()).to_bytes();
    modulus[0] += 1; // The odd modulus's predecessor has an even low byte.
    for bytes in [modulus, [0xff; 32]] {
        assert_eq!(
            <Fr as Serializable<32>>::from_bytes(&bytes),
            Err(Error::InvalidData)
        );
    }
}

#[test]
fn limb_shifts_match_bitwise_reference() {
    for scalar in [Fr::zero(), Fr::one(), -Fr::one(), Fr::from(u64::MAX)] {
        for shift in [
            0,
            1,
            63,
            64,
            65,
            127,
            128,
            129,
            191,
            192,
            193,
            255,
            256,
            257,
            u32::MAX,
        ] {
            let mut expected = [0u64; 4];
            for bit in 0..256usize {
                let source = bit as u64 + u64::from(shift);
                if source < 256 {
                    expected[bit / 64] |=
                        ((scalar[source as usize / 64] >> (source % 64)) & 1)
                            << (bit % 64);
                }
            }
            let mut actual = scalar;
            actual.divn(shift);
            for i in 0..4 {
                assert_eq!(actual[i], expected[i], "shift {shift}, limb {i}");
            }
        }
    }
}

fn group_encoding<G: Group + GroupEncoding>() {
    let random = G::random(XorShiftRng::from_seed([42; 16]));
    assert!(!bool::from(random.is_identity()));
    for point in [G::identity(), G::generator(), -G::generator(), random] {
        let bytes = point.to_bytes();
        assert_eq!(G::from_bytes(&bytes).unwrap(), point);
        assert_eq!(G::from_bytes_unchecked(&bytes).unwrap(), point);
        assert_eq!(point.double(), point + point);
    }
    let mut invalid = G::Repr::default();
    invalid.as_mut().fill(0xff);
    assert!(bool::from(G::from_bytes(&invalid).is_none()));
}

#[test]
fn group_trait_encodings() {
    group_encoding::<ExtendedPoint>();
    group_encoding::<SubgroupPoint>();
    // (u=0, v=-1) is on-curve but has order two. Only the subgroup API rejects
    // it.
    let torsion =
        AffinePoint::from_raw_unchecked(Fq::zero(), -Fq::one()).to_bytes();
    assert!(bool::from(
        <ExtendedPoint as GroupEncoding>::from_bytes(&torsion).is_some()
    ));
    assert!(bool::from(
        <SubgroupPoint as GroupEncoding>::from_bytes(&torsion).is_none()
    ));
    assert!(bool::from(
        <SubgroupPoint as GroupEncoding>::from_bytes_unchecked(&torsion)
            .is_some()
    ));
}

#[test]
fn hash_inputs_ignore_projective_scaling() {
    for original in [
        ExtendedPoint::identity(),
        GENERATOR_EXTENDED,
        -GENERATOR_EXTENDED,
    ] {
        let affine = AffinePoint::from(original);
        let [u, v] = [affine.get_u(), affine.get_v()];
        assert_eq!(ExtendedPoint::from_affine(affine), original);
        let bytes = <AffinePoint as Serializable<32>>::to_bytes(&affine);
        assert_eq!(bytes, affine.to_bytes());
        assert_eq!(
            <AffinePoint as Serializable<32>>::from_bytes(&bytes),
            Ok(affine)
        );
        for z in [Fq::one(), Fq::from(7)] {
            let scaled =
                ExtendedPoint::from_raw_unchecked(u * z, v * z, z, u * z, v);
            assert!(bool::from(scaled.is_on_curve()));
            assert_eq!(scaled.to_hash_inputs(), [u, v]);
        }
    }
}

#[test]
fn niels_arithmetic_matches_scalar_arithmetic() {
    use group::Curve;

    // Keep both inputs and expectations independent of Niels multiplication.
    fn reference_mul(scalar: Fr) -> ExtendedPoint {
        let mut point = GENERATOR_EXTENDED;
        let mut product = ExtendedPoint::identity();
        for byte in scalar.to_bytes() {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    product += point;
                }
                point = point.double();
            }
        }
        product
    }

    let scalars = [Fr::zero(), Fr::one(), -Fr::one(), Fr::from(7)];
    let points = scalars.map(reference_mul);
    assert_eq!(points[1], GENERATOR_EXTENDED);
    let mut affine = [AffinePoint::identity(); 4];
    <ExtendedPoint as Curve>::batch_normalize(&points, &mut affine);
    for (i, point) in points.into_iter().enumerate() {
        assert_eq!(affine[i], <ExtendedPoint as Curve>::to_affine(&point));
        for (j, scalar) in scalars.into_iter().enumerate() {
            let difference = reference_mul(scalars[i] - scalar);
            assert_eq!(point - points[j], difference);
            assert_eq!(point - affine[j], difference);
            assert_eq!(point - points[j].to_niels(), difference);
            assert_eq!(point - affine[j].to_niels(), difference);
            let product = reference_mul(scalars[i] * scalar);
            let mut bits = scalar.to_bytes();
            bits[31] |= 0xf0; // multiply_bits deliberately ignores these bits.
            assert_eq!(point.to_niels().multiply_bits(&bits), product);
            assert_eq!(affine[i].to_niels().multiply_bits(&bits), product);
        }
    }
}

#[test]
fn scalar_field_traits() {
    use ff::{Field, PrimeField};
    use Fr as F;

    for value in [F::ZERO, F::ONE, -F::ONE, F::from(7)] {
        assert_eq!(F::from_repr(value.to_repr()).unwrap(), value);
        assert_eq!(bool::from(value.is_odd()), value.to_bytes()[0] & 1 != 0);
        assert_eq!(Field::double(&value), value + value);
        let square = Field::square(&value);
        assert_eq!(square, value * value);
        assert_eq!(Field::sqrt(&square).unwrap().square(), square);
        let inverse = Field::invert(&value);
        assert_eq!(bool::from(inverse.is_some()), value != F::ZERO);
        if value != F::ZERO {
            assert_eq!(value * inverse.unwrap(), F::ONE);
        }
    }
    assert!(bool::from(F::from_repr([0xff; 32]).is_none()));
    for (num, div, square) in [
        (F::ZERO, F::ZERO, true),
        (F::ZERO, F::ONE, true),
        (F::ONE, F::ZERO, false),
        (F::from(4), F::from(9), true),
        (F::MULTIPLICATIVE_GENERATOR, F::ONE, false),
    ] {
        let (valid, root) = F::sqrt_ratio(&num, &div);
        assert_eq!(bool::from(valid), square);
        if square {
            assert_eq!(root.square() * div, num);
        }
        if num == F::ZERO || div == F::ZERO {
            assert_eq!(root, F::ZERO);
        }
    }
}
