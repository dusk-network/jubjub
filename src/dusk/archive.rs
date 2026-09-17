// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use dusk_bytes::Error;

use crate::{AffinePoint, ExtendedPoint, SubgroupPoint};

macro_rules! checked_archive {
    ($type:ty, $valid:expr) => {
        impl $type {
            /// Decode an untrusted archive with coordinate and point
            /// validation. Requires `rkyv-validation`; invalid data returns
            /// `InvalidData`.
            ///
            /// Affine/extended points retain full-curve semantics (including
            /// torsion); `SubgroupPoint` additionally enforces subgroup
            /// membership. Identity is valid. Protocol-specific nonidentity
            /// requirements must be checked by the caller. Existing
            /// structural `CheckBytes`, layouts and coordinates are
            /// unchanged; no normalization occurs on return.
            ///
            /// Generic `rkyv::from_bytes::<Self>` and
            /// `rkyv::check_archived_root::<Self>` validate representation
            /// only, not curve or subgroup invariants. They require a
            /// trusted source or separate semantic checks before use.
            pub fn from_archive_bytes(bytes: &[u8]) -> Result<Self, Error> {
                let point = rkyv::from_bytes::<Self>(bytes)
                    .map_err(|_| Error::InvalidData)?;
                if ($valid)(&point) {
                    Ok(point)
                } else {
                    Err(Error::InvalidData)
                }
            }
        }
    };
}

checked_archive!(AffinePoint, |point: &AffinePoint| bool::from(
    point.is_on_curve()
));
checked_archive!(ExtendedPoint, |point: &ExtendedPoint| bool::from(
    point.is_on_curve()
));
checked_archive!(SubgroupPoint, |point: &SubgroupPoint| {
    // Do not run group arithmetic on invalid projective coordinates.
    bool::from(point.0.is_on_curve()) && bool::from(point.0.is_torsion_free())
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fq, Fr, GENERATOR_EXTENDED};

    #[test]
    fn strict_archives_preserve_full_curve_and_subgroup_distinction() {
        for point in [
            ExtendedPoint::identity(),
            GENERATOR_EXTENDED * Fr::from(7u64),
        ] {
            let bytes = rkyv::to_bytes::<_, 256>(&point).unwrap();
            let reopened = ExtendedPoint::from_archive_bytes(&bytes).unwrap();
            assert_eq!(
                bytes.as_slice(),
                rkyv::to_bytes::<_, 256>(&reopened).unwrap().as_slice()
            );
            assert!(ExtendedPoint::from_archive_bytes(&bytes[..1]).is_err());
            let subgroup = SubgroupPoint(point);
            let bytes = rkyv::to_bytes::<_, 256>(&subgroup).unwrap();
            assert!(SubgroupPoint::from_archive_bytes(&bytes).is_ok());
        }
        let torsion = AffinePoint::from_raw_unchecked(Fq::zero(), -Fq::one());
        let bytes = rkyv::to_bytes::<_, 256>(&torsion).unwrap();
        assert!(AffinePoint::from_archive_bytes(&bytes).is_ok());
        let extended = ExtendedPoint::from(torsion);
        let bytes = rkyv::to_bytes::<_, 256>(&extended).unwrap();
        assert!(ExtendedPoint::from_archive_bytes(&bytes).is_ok());
        let bytes = rkyv::to_bytes::<_, 256>(&SubgroupPoint(extended)).unwrap();
        assert!(rkyv::from_bytes::<SubgroupPoint>(&bytes).is_ok());
        assert!(SubgroupPoint::from_archive_bytes(&bytes).is_err());

        let zero = Fq::zero();
        let mut bad_t = GENERATOR_EXTENDED;
        bad_t.t1 = -bad_t.t1;
        for point in [
            ExtendedPoint::from_raw_unchecked(zero, zero, zero, zero, zero),
            bad_t,
        ] {
            let bytes = rkyv::to_bytes::<_, 256>(&point).unwrap();
            assert!(rkyv::from_bytes::<ExtendedPoint>(&bytes).is_ok());
            assert!(ExtendedPoint::from_archive_bytes(&bytes).is_err());
            let bytes =
                rkyv::to_bytes::<_, 256>(&SubgroupPoint(point)).unwrap();
            assert!(SubgroupPoint::from_archive_bytes(&bytes).is_err());
        }
        let off_curve = AffinePoint::from_raw_unchecked(zero, zero);
        let bytes = rkyv::to_bytes::<_, 256>(&off_curve).unwrap();
        assert!(rkyv::from_bytes::<AffinePoint>(&bytes).is_ok());
        assert!(AffinePoint::from_archive_bytes(&bytes).is_err());
        // Representation checks must run before semantic arithmetic.
        let mut bytes =
            rkyv::to_bytes::<_, 256>(&AffinePoint::identity()).unwrap();
        bytes[..32].fill(0xff);
        assert!(AffinePoint::from_archive_bytes(&bytes).is_err());
    }
}
