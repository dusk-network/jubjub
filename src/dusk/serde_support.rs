use core::fmt;

use dusk_bytes::Serializable;
use serde::de::{Error, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{AffinePoint, ExtendedPoint};

/// Decodes a hex string of exactly `N` bytes into a stack buffer, checking
/// the encoded length before decoding any characters.
pub(crate) fn deserialize_hex<'de, D: Deserializer<'de>, const N: usize>(
    deserializer: D,
) -> Result<[u8; N], D::Error> {
    struct Hex<const N: usize>;

    impl<'de, const N: usize> Visitor<'de> for Hex<N> {
        type Value = [u8; N];

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "a hex string encoding {N} bytes")
        }

        fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
            self.visit_bytes(value.as_bytes())
        }

        fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<Self::Value, E> {
            let mut bytes = [0; N];
            hex::decode_to_slice(value, &mut bytes).map_err(E::custom)?;
            Ok(bytes)
        }
    }

    deserializer.deserialize_str(Hex::<N>)
}

impl Serialize for AffinePoint {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let s = hex::encode(self.to_bytes());
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for AffinePoint {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let bytes: [u8; Self::SIZE] = deserialize_hex(deserializer)?;
        AffinePoint::from_bytes(bytes).into_option().ok_or_else(|| {
            Error::custom(
                "Failed to deserialize AffinePoint: invalid AffinePoint",
            )
        })
    }
}

impl Serialize for ExtendedPoint {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        AffinePoint::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExtendedPoint {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        AffinePoint::deserialize(deserializer).map(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::boxed::Box;

    use group::Group;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::{dusk::test_utils, AffinePoint, ExtendedPoint};

    #[test]
    fn serde_affine_point() -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(0xdead);
        let point = ExtendedPoint::random(&mut rng);
        let point = AffinePoint::from(point);
        let ser = test_utils::assert_canonical_json(
            &point,
            "\"7bdf072820ef769376583c858687144b0dcaccf8319880627c82eef222f8c6cb\""
        )?;
        let deser = serde_json::from_str(&ser)?;
        assert_eq!(point, deser);
        Ok(())
    }

    #[test]
    fn serde_extended_point() -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(0xdead);
        let point = ExtendedPoint::random(&mut rng);
        let ser = test_utils::assert_canonical_json(
            &point,
            "\"7bdf072820ef769376583c858687144b0dcaccf8319880627c82eef222f8c6cb\""
        )?;
        let deser = serde_json::from_str(&ser)?;
        assert_eq!(point, deser);
        Ok(())
    }

    #[test]
    fn serde_affine_point_wrong_encoded() {
        let wrong_encoded = "wrong-encoded";

        let affine_point: Result<AffinePoint, _> =
            serde_json::from_str(wrong_encoded);
        assert!(affine_point.is_err());
    }

    #[test]
    fn serde_affine_point_too_long_encoded() {
        let length_33_enc = "\"e4ab9de40283a85d6ea0cd0120500697d8b01c71b7b4b520292252d20937000631\"";

        let affine_point: Result<AffinePoint, _> =
            serde_json::from_str(length_33_enc);
        assert!(affine_point.is_err());
    }

    #[test]
    fn serde_affine_point_too_short_encoded() {
        let length_31_enc = "\"1751c37a1dca7aa4c048fcc6177194243edc3637bae042e167e4285945e046\"";

        let affine_point: Result<AffinePoint, _> =
            serde_json::from_str(length_31_enc);
        assert!(affine_point.is_err());
    }
}
