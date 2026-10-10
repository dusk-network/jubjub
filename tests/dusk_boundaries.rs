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
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

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

#[test]
fn point_negation_and_sum() {
    let p = GENERATOR_EXTENDED * Fr::from(7u64);
    let q = GENERATOR_EXTENDED * Fr::from(11u64);
    let affine = AffinePoint::from(p);

    // -(u, v) = (-u, v)
    assert_eq!((-affine).get_u(), -affine.get_u());
    assert_eq!((-affine).get_v(), affine.get_v());
    assert_eq!(ExtendedPoint::from(-affine), -p);
    for neg in [-p, ExtendedPoint::from(-affine)] {
        assert!(bool::from(neg.is_on_curve()));
        assert_eq!(p + neg, ExtendedPoint::identity());
        assert_eq!(q + neg, q - p);
    }

    assert_eq!([p, q].into_iter().sum::<ExtendedPoint>(), p + q);
    assert_eq!([p, q, -p].iter().sum::<ExtendedPoint>(), q);
    assert_eq!(
        core::iter::empty::<ExtendedPoint>().sum::<ExtendedPoint>(),
        ExtendedPoint::identity()
    );
}

#[test]
fn point_equality_compares_both_coordinates() {
    let p = GENERATOR_EXTENDED * Fr::from(7u64);
    let affine = AffinePoint::from(p);
    let (u, v) = (affine.get_u(), affine.get_v());
    assert_eq!(p, ExtendedPoint::from(affine));

    // Both are on the curve, and each shares one coordinate with `p`.
    for other in [
        AffinePoint::from_raw_unchecked(-u, v),
        AffinePoint::from_raw_unchecked(u, -v),
    ] {
        assert!(bool::from(other.is_on_curve()));
        assert!(!bool::from(affine.ct_eq(&other)));
        assert_ne!(affine, other);
        assert!(!bool::from(p.ct_eq(&ExtendedPoint::from(other))));
        assert_ne!(p, ExtendedPoint::from(other));
    }
}

#[test]
fn prime_order_excludes_identity_and_torsion() {
    // (u=0, v=-1) has order two.
    let torsion = ExtendedPoint::from(AffinePoint::from_raw_unchecked(
        Fq::zero(),
        -Fq::one(),
    ));
    let p = GENERATOR_EXTENDED * Fr::from(7u64);

    for (point, prime_order) in [
        (ExtendedPoint::identity(), false),
        (torsion, false),
        (p + torsion, false),
        (p, true),
    ] {
        assert_eq!(bool::from(point.is_prime_order()), prime_order);
        assert_eq!(
            bool::from(AffinePoint::from(point).is_prime_order()),
            prime_order
        );
    }
}

#[test]
fn subgroup_and_mixed_operators_match_scalar_multiplication() {
    let g = SubgroupPoint::generator();
    let at = |scalar: Fr| ExtendedPoint::from(g) * scalar;
    let (a, b) = (Fr::from(7u64), Fr::from(11u64));
    let (p, q) = (g * a, g * b);
    assert_eq!(ExtendedPoint::from(p), at(a));

    let subgroup = [
        (p + q, a + b),
        (p - q, a - b),
        (-p, -a),
        (-&p, -a),
        ([p, q].iter().sum(), a + b),
        (
            SubgroupPoint::conditional_select(&p, &q, Choice::from(0)),
            a,
        ),
        (
            SubgroupPoint::conditional_select(&p, &q, Choice::from(1)),
            b,
        ),
    ];
    for (point, scalar) in subgroup {
        assert_eq!(ExtendedPoint::from(point), at(scalar));
    }

    let (extended, affine) = (at(a), AffinePoint::from(at(b)));
    assert_eq!(extended + q, at(a + b));
    assert_eq!(extended - q, at(a - b));
    assert_eq!(extended + affine, at(a + b));
    assert_eq!(extended - affine, at(a - b));
    assert_eq!(affine * a, at(a * b));
}

#[test]
fn group_generators() {
    // The full group's generator has neither small nor prime order, and its
    // cofactor multiple generates the prime-order subgroup.
    let generator = ExtendedPoint::generator();
    assert!(bool::from(generator.is_on_curve()));
    assert!(!bool::from(generator.is_small_order()));
    assert!(!bool::from(generator.is_torsion_free()));

    let subgroup = ExtendedPoint::from(SubgroupPoint::generator());
    assert!(bool::from(subgroup.is_prime_order()));
    assert_eq!(subgroup, generator.mul_by_cofactor());
}

#[test]
fn affine_group_encoding() {
    let random = ExtendedPoint::random(XorShiftRng::from_seed([42; 16]));
    for point in [
        AffinePoint::identity(),
        AffinePoint::from(GENERATOR_EXTENDED),
        AffinePoint::from(random),
    ] {
        let bytes = <AffinePoint as GroupEncoding>::to_bytes(&point);
        assert_eq!(bytes, point.to_bytes());
        assert_eq!(
            <AffinePoint as GroupEncoding>::from_bytes(&bytes).unwrap(),
            point
        );
        assert_eq!(
            <AffinePoint as GroupEncoding>::from_bytes_unchecked(&bytes)
                .unwrap(),
            point
        );
    }
}

/// The scalar whose internal Montgomery limbs are `limbs`, which must be below
/// the modulus.
fn from_montgomery_limbs(limbs: [u64; 4]) -> Fr {
    let scalar = Fr::from_raw(limbs).reduce();
    for (i, limb) in limbs.into_iter().enumerate() {
        assert_eq!(scalar[i], limb);
    }
    scalar
}

#[test]
fn scalar_equality_compares_every_limb() {
    let base = [5, 6, 7, 8];
    let scalar = from_montgomery_limbs(base);
    for limb in 0..4 {
        let mut limbs = base;
        limbs[limb] += 1;
        let other = from_montgomery_limbs(limbs);
        assert!(!bool::from(scalar.ct_eq(&other)));
        assert_ne!(scalar, other);
    }
    assert_eq!(scalar, from_montgomery_limbs(base));
}

#[test]
fn scalar_negation_of_sparse_limbs() {
    // Every nonzero pattern of zero and nonzero limbs.
    for pattern in 1..16 {
        let limbs: [u64; 4] = core::array::from_fn(|i| 5 * (pattern >> i & 1));
        let scalar = from_montgomery_limbs(limbs);
        assert_ne!(-scalar, Fr::zero());
        assert_eq!(-scalar, Fr::zero() - scalar);
        assert_eq!(scalar + -scalar, Fr::zero());
    }
    assert_eq!(-Fr::zero(), Fr::zero());
}

#[test]
fn scalar_sum_and_product() {
    let scalars = [Fr::from(3u64), Fr::from(5u64), -Fr::from(7u64)];
    assert_eq!(scalars.iter().sum::<Fr>(), Fr::one());
    assert_eq!(scalars.into_iter().product::<Fr>(), -Fr::from(105u64));
    assert_eq!(core::iter::empty::<Fr>().sum::<Fr>(), Fr::zero());
    assert_eq!(core::iter::empty::<Fr>().product::<Fr>(), Fr::one());
}

#[test]
fn scalar_conversions_and_bits() {
    use ff::PrimeField;

    let scalar = -Fr::from(7u64);
    assert_eq!(<[u8; 32]>::from(scalar), scalar.to_bytes());
    assert_eq!(<[u8; 32]>::from(&scalar), scalar.to_bytes());
    assert_eq!(Fr::NUM_BITS, 252);
    assert_eq!(Fr::CAPACITY, 251);

    #[cfg(feature = "bits")]
    {
        use ff::PrimeFieldBits;

        let mut modulus = (-Fr::one()).to_bytes();
        modulus[0] += 1; // The odd modulus's predecessor has an even low byte.
        for (bits, bytes) in [
            (scalar.to_le_bits(), scalar.to_bytes()),
            (Fr::char_le_bits(), modulus),
        ] {
            for (i, bit) in bits.iter().enumerate() {
                assert_eq!(*bit, bytes[i / 8] >> (i % 8) & 1 == 1, "bit {i}");
            }
        }
    }
}

#[test]
fn scalar_order_compares_limbs_most_significant_first() {
    use core::cmp::Ordering;

    let low = from_montgomery_limbs([6, 0, 0, 0]);
    for (a, b, order) in [
        (low, low, Ordering::Equal),
        (from_montgomery_limbs([5, 0, 0, 0]), low, Ordering::Less),
        (from_montgomery_limbs([0, 0, 0, 1]), low, Ordering::Greater),
        (from_montgomery_limbs([7, 0, 0, 0]), low, Ordering::Greater),
        // Each pair differs only in a middle limb.
        (from_montgomery_limbs([6, 1, 0, 0]), low, Ordering::Greater),
        (from_montgomery_limbs([6, 0, 1, 0]), low, Ordering::Greater),
        // A middle limb outranks every less significant one.
        (from_montgomery_limbs([0, 1, 0, 0]), low, Ordering::Greater),
        (
            from_montgomery_limbs([0, 0, 1, 0]),
            from_montgomery_limbs([6, 5, 0, 0]),
            Ordering::Greater,
        ),
    ] {
        assert_eq!(a.cmp(&b), order);
        assert_eq!(b.cmp(&a), order.reverse());
        assert_eq!(a.partial_cmp(&b), Some(order));
    }
}

#[cfg(feature = "alloc")]
#[test]
fn wnaf_with_recommended_windows() {
    let scalar = Fr::hash_to_scalar(None, b"wnaf");
    let expected = GENERATOR_EXTENDED * scalar;
    for num_scalars in [1, 2, 64, 4096] {
        let mut wnaf = group::Wnaf::new();
        let point = wnaf.base(GENERATOR_EXTENDED, num_scalars).scalar(&scalar);
        assert_eq!(point, expected, "{num_scalars} scalars");
    }
}
