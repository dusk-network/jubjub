/// This file implements the elgamal encryption 
/// The implementation uses the JubJub elliptic curve

use crate::{GENERATOR, AffinePoint, ExtendedPoint, ExtendedNielsPoint, AffineNielsPoint, Fq};
use crate::fr::Fr;
use subtle::ConstantTimeEq;
use crate::Error;
use rand::Rng;

pub struct SecretKey(Fr);

const Q: [u8;32] = [14, 125, 180, 234, 101, 51, 175, 169, 6, 103, 59, 1, 1, 52, 59, 0, 166, 104, 32, 147, 204, 200, 16, 130, 208, 151, 14, 94, 214, 247, 44, 183];
const R: [u8;32] = [115, 237, 167, 83, 41, 157, 125, 72, 51, 57, 216, 8, 9, 161, 216, 5, 83, 189, 164, 2, 255, 254, 91, 254, 255, 255, 255, 255, 0, 0, 0, 1];

impl SecretKey {
    // This will create a new private key
    // from a scalar of the Field Fr.
    pub fn new() -> Result<SecretKey, Error> {
        let scalar = random_fr();
        if scalar.ct_eq(&Fr::zero()).unwrap_u8() == 1u8 {
            return Err(Error::InvalidParameters);
        }

        Ok(SecretKey(scalar))
    }

    /// `to_public` returns the `PublicKey` of the `PrivateKey`.
    pub fn to_public(&self) -> PublicKey {
        let point = AffinePoint::from(ExtendedPoint::from(GENERATOR) * &self.0);
        PublicKey(point)
    }
}


pub struct PublicKey(AffinePoint);

impl PublicKey {
    // This will create a new public key from a 
    // secret key
    pub fn from_secret(secret: &SecretKey) -> PublicKey {
        let point = AffinePoint::from(ExtendedPoint::from(GENERATOR) * secret.0);

        PublicKey(point)
    }
    
    pub fn new() -> Result<PublicKey, Error> {
        let sk = SecretKey::new();
        let pk = SecretKey::to_public(&sk.unwrap());
        Ok(pk)
    }
}

pub struct KeyPair {
    secret_key: SecretKey,
    public_key: PublicKey,
}

// impl KeyPair {
//     pub fn new() -> Result<KeyPair, Error> {
//         let secret_key = SecretKey::new();
//         let public_key = private_key.from_secret();

//         let keys = KeyPair { secret_key, public_key };
        
//         Ok(keys)
//     }
// }

pub struct CipherText {
    publicKey: PublicKey,
    cipher: AffinePoint,
}

// impl CipherText {

//     pub fn random() -> Result<CipherText, Error> {
//         let public = PublicKey::new();
//         let x = SecretKey::new();
//         let y = SecretKey::new();
//         let cipher = from_raw_unchecked(x, y);

//         let cyph = CypherText { publicKey, cipher };
//         Ok(cyph)
//     }
// }


#[derive(Debug, PartialEq, Eq)]
pub struct Message([u8; 32]);

impl Message {
    pub fn random() -> Message {
        let mut rng = rand::thread_rng();
        let n1: u64 = rng.gen();
        let n2: u64 = rng.gen();
        let n3: u64 = rng.gen();
        let n4: u64 = rng.gen();
        let m = Fr::from_raw([n1, n2, n3, n4]);
        Message(m.to_bytes())
    }
}


/// By means of Elgamal, this function uses a 
/// KeyPair to convert a message into Cipher 
/// text. 

pub fn encrypt(sk: SecretKey, pk: PublicKey, msg: &Message) -> CipherText {
    let s = (ExtendedPoint::from(pk.0) * sk.0);
    let M = s * Fr::from_bytes(&msg.0).unwrap();

    CipherText {
        publicKey: sk.to_public(), 
        cipher: AffinePoint::from(M),
    }
}

pub fn decrypt(sk: SecretKey, cipher: CipherText) -> Message {
    let s_inv = ExtendedPoint::from(cipher.publicKey.0) * (Fr::from_bytes(&Q).unwrap() - sk.0);

    let m = ExtendedPoint::from(cipher.cipher) + s_inv;
    Message(AffinePoint::from(m).to_bytes())
}

fn random_fr() -> Fr {
    let mut rng = rand::thread_rng();
    let n1: u64 = rng.gen();
    let n2: u64 = rng.gen();
    let n3: u64 = rng.gen();
    let n4: u64 = rng.gen();
    Fr::from_raw([n1, n2, n3, n4])
}

// fn euclid_algorithm<T: Copy + Integer>(a: T, b: T) -> (T, T, T) {
//     assert!(a < b);
//     if a == T::zero() {
//         return (b, T::zero(), T::one());
//     }
//     else {
//         let (g, x, y) = egcd(b % a, a);
//         return (g, y - (b / a) * x, x);
//     }
// }

// pub fn mod_inv<T: Copy + Integer>(a: FR, &Q) -> Option<T> {
//     let (g, x, _) = euclid_algorithm(a, &Q);
//     if g != T::one() {
//         return None;
//     }
//     else {
//         return Some(x % m);
//     }
// }

#[test]
fn test_encryption() {
    for _ in 0..10 {
        let msg1 = Message::random();
        let sk1 = SecretKey::new().unwrap();
        let sk2 = SecretKey::new().unwrap();
        let pk2 = PublicKey::from_secret(&sk2);

        let cyph = encrypt(sk1, pk2, &msg1);
        let msg2 = decrypt(sk2, cyph);

        assert_eq!(msg1, msg2);
    }
}






