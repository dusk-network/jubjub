extern crate alloc;

use alloc::string::{String, ToString};

use dusk_bytes::Serializable;
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

use crate::{AffinePoint, ExtendedPoint};

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
        let s = String::deserialize(deserializer)?;
        let decoded = hex::decode(&s).map_err(Error::custom)?;
        let decoded_len = decoded.len();
        let bytes: [u8; Self::SIZE] = decoded.try_into().map_err(|_| {
            Error::invalid_length(decoded_len, &Self::SIZE.to_string().as_str())
        })?;
        AffinePoint::from_bytes(bytes)
            .into_option()
            .ok_or(Error::custom(
                "Failed to deserialize AffinePoint: invalid AffinePoint",
            ))
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
    use group::Group;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::{AffinePoint, ExtendedPoint};

    #[test]
    fn serde_affine_point() {
        let mut rng = StdRng::seed_from_u64(0xdead);
        let point = ExtendedPoint::random(&mut rng);
        let point = AffinePoint::from(point);
        let ser = serde_json::to_string(&point).unwrap();
        let deser = serde_json::from_str(&ser).unwrap();
        assert_eq!(point, deser);
    }

    #[test]
    fn serde_wrong_encoded() {
        let wrong_encoded = "wrong-encoded";

        let affine_point: Result<AffinePoint, _> =
            serde_json::from_str(&wrong_encoded);
        assert!(affine_point.is_err());
    }

    #[test]
    fn serde_too_long_encoded() {
        let length_33_enc = "\"e4ab9de40283a85d6ea0cd0120500697d8b01c71b7b4b520292252d20937000631\"";

        let affine_point: Result<AffinePoint, _> =
            serde_json::from_str(&length_33_enc);
        assert!(affine_point.is_err());
    }

    #[test]
    fn serde_too_short_encoded() {
        let length_31_enc = "\"1751c37a1dca7aa4c048fcc6177194243edc3637bae042e167e4285945e046\"";

        let affine_point: Result<AffinePoint, _> =
            serde_json::from_str(&length_31_enc);
        assert!(affine_point.is_err());
    }
}
