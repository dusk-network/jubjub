use dusk_jubjub::{
    AffineNielsPoint, AffinePoint, ExtendedNielsPoint, ExtendedPoint, Fr,
    SubgroupPoint, GENERATOR_EXTENDED,
};

#[test]
fn archive_layout_and_roundtrip() {
    let scalar = Fr::from(42u64);
    let point = GENERATOR_EXTENDED * Fr::from(7u64);
    let affine = AffinePoint::from(point);
    // Keep fixture inputs independent of multiplication's projective form.
    let point = ExtendedPoint::from(affine);
    let subgroup =
        SubgroupPoint::from_raw_unchecked(affine.get_u(), affine.get_v());
    // Little-/big-endian BLAKE2b-512 digests below were generated at
    // 0fef247db9a9199980884058bfe0f103efd903ed, before custom checks.
    // Detect the selected archive byte order, which need not match the host.
    let little_endian = rkyv::to_bytes::<_, 8>(&1u64).unwrap()[0] == 1;
    macro_rules! check {
        ($type:ty, $value:expr, $size:expr, $le:expr, $be:expr) => {{
            let bytes = rkyv::to_bytes::<_, 256>(&$value).unwrap();
            let expected = if little_endian { $le } else { $be };
            assert_eq!(
                blake2b_simd::blake2b(&bytes).to_hex().as_str(),
                expected,
                "{} archive layout changed",
                stringify!($type)
            );
            assert_eq!(bytes.len(), $size);
            assert_eq!(size_of::<rkyv::Archived<$type>>(), $size);
            let reopened = rkyv::from_bytes::<$type>(&bytes).unwrap();
            assert_eq!(
                bytes.as_slice(),
                rkyv::to_bytes::<_, 256>(&reopened).unwrap().as_slice()
            );
            assert!(
                rkyv::from_bytes::<$type>(&bytes[..bytes.len() - 1]).is_err()
            );
            reopened
        }};
    }
    let reopened = check!(
        Fr,
        scalar,
        32,
        "3099a68b429d812c12575e3ce307b65f6105f87db90722b9ca769393f7dc1960\
         755be2895102cf2c88f5082f1807a7af8115cdb623bc9f0581bcbc6d7a483bec",
        "67f4b792314799d779395018139fcb59b271a7812868883bf5fdbcf4cdaed42e\
         225d5a40e86b3a4c41401616e3e526c618febcbaaafc6a4a130ae0941cecb786"
    );
    assert_eq!(reopened, scalar);
    let reopened = check!(
        AffinePoint,
        affine,
        64,
        "a7814289fcb1e88a05fc58ddd2915c63c971166798a0bf25a0d85e91acc9f290\
         2e9e66f6e417820ef736a87eaa1b1317e479c48d411858517e16c8a008e890d3",
        "397faf54eb818ccc5887586c817d802011a306404315ea69e6462a548e43d89e\
         bc13776f8dcb46b85833dfea87abb13e55555acbac1b0943ccdf9a4460da2538"
    );
    assert_eq!(reopened, affine);
    let reopened = check!(
        ExtendedPoint,
        point,
        160,
        "ba9de9a1c7afe6c078906db739fcf259352870868af4f75039ace40deaa90ef1\
         5ff7ed263d1ba062729d2efac048886d85feeb30cd0f9ca965b663b34c3d95e9",
        "29d67b60155aca82a736af81c46f53c46784ac44d48babbf39d0f39ce461b709\
         bf6bbdd3facc3a283235646af30f452d575c9bdb2ad00ea9194851d674bb945f"
    );
    assert_eq!(reopened, point);
    assert!(bool::from(reopened.is_on_curve()));
    let reopened = check!(
        AffineNielsPoint,
        affine.to_niels(),
        96,
        "7085bb8bc0ac33f13c332ad2dc645d2ed71f9ff5b9bdcbf8d6eefdf840d03286\
         97607c5fa6384e58d397a0d7674dd1347752259ed7b0b19ee334ea9528ad0071",
        "6b7a59f5604bd9a25e999dbfb3e3a883ff878a1774f938c45b55ab9ae9169412\
         3890fbb970c76f01e562d61888b9e3bef519014a8a44e4dde5e6d00289acfe7f"
    );
    assert_eq!(ExtendedPoint::identity() + reopened, point);
    let reopened = check!(
        ExtendedNielsPoint,
        point.to_niels(),
        128,
        "85c5fa0dd6ef7ba88d15fcfdbe2c03bb646d0e7165659e82229ca5ca285aa6fa\
         39f225d9e5166411dcc4d1ab9fba8a59cd2c9aa3ba67c48c99dc1ed9cd4c2c03",
        "db290886602a8bd2d7833c43d4b3deadf076373cd56d12ff665bbe801a8fbf05\
         d618757833891e7d4da3b0f59269d22c36d61aec6f7df080dddd2a9e99c28ea8"
    );
    assert_eq!(ExtendedPoint::identity() + reopened, point);
    let reopened = ExtendedPoint::from(check!(
        SubgroupPoint,
        subgroup,
        160,
        "ba9de9a1c7afe6c078906db739fcf259352870868af4f75039ace40deaa90ef1\
         5ff7ed263d1ba062729d2efac048886d85feeb30cd0f9ca965b663b34c3d95e9",
        "29d67b60155aca82a736af81c46f53c46784ac44d48babbf39d0f39ce461b709\
         bf6bbdd3facc3a283235646af30f452d575c9bdb2ad00ea9194851d674bb945f"
    ));
    assert_eq!(reopened, point);
    assert!(bool::from(reopened.is_torsion_free()));
}

#[test]
fn scalar_archive_requires_canonical_montgomery_limbs() {
    let modulus = [
        0xd097_0e5e_d6f7_2cb7,
        0xa668_2093_ccc8_1082,
        0x0667_3b01_0134_3b00,
        0x0e7d_b4ea_6533_afa9,
    ];
    let mut below = modulus;
    below[0] -= 1;
    for limbs in [[0; 4], [1, 0, 0, 0], below] {
        let bytes = rkyv::to_bytes::<_, 256>(&limbs).unwrap();
        let scalar = rkyv::from_bytes::<Fr>(&bytes).unwrap();
        for i in 0..4 {
            assert_eq!(scalar[i], limbs[i]);
        }
    }
    let mut above = modulus;
    above[0] += 1;
    for limbs in [modulus, above, [u64::MAX; 4]] {
        let bytes = rkyv::to_bytes::<_, 256>(&limbs).unwrap();
        assert!(rkyv::from_bytes::<Fr>(&bytes).is_err());
        let bytes = rkyv::to_bytes::<_, 256>(&[[0u64; 4], limbs]).unwrap();
        assert!(rkyv::from_bytes::<[Fr; 2]>(&bytes).is_err());
        let bytes = rkyv::to_bytes::<_, 256>(&vec![[0u64; 4], limbs]).unwrap();
        assert!(rkyv::from_bytes::<Vec<Fr>>(&bytes).is_err());
    }
}
