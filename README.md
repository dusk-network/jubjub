# jubjub [![Crates.io](https://img.shields.io/crates/v/jubjub.svg)](https://crates.io/crates/dusk-jubjub) #

<img
 width="15%"
 align="right"
 src="https://raw.githubusercontent.com/zcash/zips/master/protocol/jubjub.png"/>

This is a pure Rust implementation of the Jubjub elliptic curve group and its associated fields.

> :warning: THIS CRATE IS A FORK OF [https://github.com/zkcrypto/jubjub](https://github.com/zkcrypto/jubjub/): The Dusk team has added a variety of features for its own use-case on the top of the original library. You SHOULD NOT use this library unless you need a specific feature that we've implemented and is not available in the original.

* **This implementation has not been reviewed or audited. Use at your own risk.**
* This implementation targets Rust `1.56` or later.
* All operations are constant time unless explicitly noted.
* This implementation does not require the Rust standard library.

## Dusk Additions

- Diffie-Hellman Key Exchange (DHKE) for Jubjub curves for secure shared secrets.
- Exposes fixed generator points.
- Enhance serialization for Jubjub affine points.
- Robust hashing mechanism to map bytes to a point on the Jubjub curve through rejection sampling
- Bitwise shifts and reductions for arithmatic within the scalar field.
- wnaf implementation for scalar multiplication.
- Comparative and ordinal operations for scalars, for sorting and equality checks.
- Scalar generation from bytes using BLAKE2b hashing.
- Provide `serde` feature for opinionated de- & serialization of `Fr`, `AffinePoint` and `ExtendedPoint` types as hex-encoded bytes.


## [Documentation](https://docs.rs/dusk-jubjub/)

## Curve Description

Jubjub is the [twisted Edwards curve](https://en.wikipedia.org/wiki/Twisted_Edwards_curve) `-u^2 + v^2 = 1 + d.u^2.v^2` of rational points over `GF(q)` with a subgroup of prime order `r` and cofactor `8`.

```
q = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
r = 0x0e7db4ea6533afa906673b0101343b00a6682093ccc81082d0970e5ed6f72cb7
d = -(10240/10241)
```

The choice of `GF(q)` is made to be the scalar field of the BLS12-381 elliptic curve construction.

Jubjub is birationally equivalent to a [Montgomery curve](https://en.wikipedia.org/wiki/Montgomery_curve) `y^2 = x^3 + Ax^2 + x` over the same field with `A = 40962`. This value of `A` is the smallest integer such that `(A - 2) / 4` is a small integer, `A^2 - 4` is nonsquare in `GF(q)`, and the Montgomery curve and its quadratic twist have small cofactors `8` and `4`, respectively. This is identical to the relationship between Curve25519 and ed25519.

Please see [./doc/evidence/](./doc/evidence/) for supporting evidence that Jubjub meets the [SafeCurves](https://safecurves.cr.yp.to/index.html) criteria. The tool in [./doc/derive/](./doc/derive/) will derive the curve parameters via the above criteria to demonstrate rigidity.

## Acknowledgements

Jubjub was designed by Sean Bowe. Daira Hopwood is responsible for its name and specification. The security evidence in [./doc/evidence/](./doc/evidence/) is the product of Daira Hopwood and based on SafeCurves by Daniel J. Bernstein and Tanja Lange. Peter Newell and Daira Hopwood are responsible for the Jubjub bird image.

Please see `Cargo.toml` for a list of primary authors of this codebase.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
