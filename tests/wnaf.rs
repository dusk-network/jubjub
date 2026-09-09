use dusk_jubjub::Fr;

#[test]
fn wnaf_reconstructs_the_integer() {
    let mut scalars = vec![Fr::zero(), Fr::one()];
    scalars.extend((1..=128).map(|i| -Fr::from(i)));
    for bit in [63, 64, 127, 128, 191, 192, 251] {
        let mut limbs = [0; 4];
        limbs[bit / 64] = 1 << (bit % 64);
        let scalar = Fr::from_raw(limbs);
        scalars.extend([scalar - Fr::one(), scalar, scalar + Fr::one()]);
    }
    scalars.extend(
        (0u64..256).map(|i| Fr::hash_to_scalar(None, &i.to_le_bytes())),
    );
    for scalar in scalars {
        for width in 2..=8 {
            let digits = scalar.compute_windowed_naf(width);
            let mut integer = [0u64; 4];
            // Horner reconstruction in base 2, using signed wide intermediates
            // rather than any field arithmetic or the recoder's limb helpers.
            for &digit in digits.iter().rev() {
                let mut carry = i128::from(digit);
                for limb in &mut integer {
                    let value = i128::from(*limb) * 2 + carry;
                    *limb = value as u64;
                    carry = value >> 64;
                }
                assert_eq!(carry, 0);
            }
            let bytes: Vec<_> =
                integer.into_iter().flat_map(u64::to_le_bytes).collect();
            assert_eq!(bytes, scalar.to_bytes());
            let mut next = 0;
            for (i, digit) in digits.into_iter().enumerate() {
                if digit != 0 {
                    assert_eq!(digit & 1, 1);
                    assert!(i16::from(digit).abs() < 1 << (width - 1));
                    assert!(i >= next);
                    next = i + usize::from(width);
                }
            }
        }
    }
}

#[test]
fn width_two_matches_baseline() {
    let mut state = blake2b_simd::Params::new().to_state();
    for i in 0u64..256 {
        let scalar = Fr::hash_to_scalar(None, &i.to_le_bytes());
        state.update(&scalar.compute_windowed_naf(2).map(|digit| digit as u8));
    }
    // Generated at 0fef247db9a9199980884058bfe0f103efd903ed.
    assert_eq!(
        state.finalize().to_hex().as_str(),
        concat!(
            "7fbe802931c36c3d0e338092a25e43604d0f5d6543c5b47131dfb09bd157ec1ce",
            "9a067bde18aeb506f4dba84532a0cdba9dbaf23c0532d6470e74e3c0527367b"
        )
    );
}

#[test]
fn window_bounds_and_balanced_residues() {
    for width in 0..=u8::MAX {
        if !(2..=8).contains(&width) {
            for scalar in [Fr::zero(), Fr::one()] {
                assert!(std::panic::catch_unwind(
                    || scalar.compute_windowed_naf(width)
                )
                .is_err());
            }
        }
    }
    for width in 1..=8 {
        let modulus = 1i16 << width;
        for value in 0u64..256 {
            let scalar = Fr::from(value).reduce();
            let residue = value as i16 % modulus;
            let expected = (residue + modulus / 2) % modulus - modulus / 2;
            assert_eq!(i16::from(scalar.mods_2_pow_k(width)), expected);
            assert_eq!(i16::from(scalar.mod_2_pow_k(width)), residue);
        }
    }
    assert_eq!(Fr::one().mod_2_pow_k(0), 0);
    for width in [0, 9, 255] {
        assert!(std::panic::catch_unwind(|| Fr::zero().mods_2_pow_k(width))
            .is_err());
    }
    assert!(std::panic::catch_unwind(|| Fr::zero().mod_2_pow_k(9)).is_err());
}
