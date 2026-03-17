# AGENTS.md — dusk-jubjub

## Care Level: Cryptographic — Elevated

This is critical cryptographic code. A subtle bug here can break
soundness or enable fund theft. Do not introduce timing side-channels.

## Overview

JubJub elliptic curve group implementation. **Fork of zkcrypto/jubjub** with
Dusk-specific enhancements (dhke, generators, serde, rkyv).

All Dusk additions are scoped inside `dusk` submodules (e.g. `src/dusk.rs`,
`src/dusk/`, `src/fr/dusk.rs`). Do not modify the upstream zkcrypto code
directly.

## Commands

```bash
make test      # Run tests (std + no_std)
make clippy    # Run clippy
make fmt       # Format code (requires nightly)
make check     # Type-check
make doc       # Generate docs
make no-std    # Verify no_std + WASM compatibility
make clean     # Clean build artifacts
cargo bench    # Run benchmarks
```

## Architecture

### Key Files

| Path | Purpose |
|------|---------|
| `src/lib.rs` | AffinePoint, ExtendedPoint, core curve operations |
| `src/fr.rs` | Scalar field Fr implementation |
| `src/dusk.rs` | Dusk extensions: dhke, generators, additional ops |
| `src/dusk/serde_support.rs` | Serde for AffinePoint/ExtendedPoint |
| `src/fr/dusk.rs` | Dusk-specific Fr operations |
| `src/elgamal.rs` | ElGamal encryption |
| `src/util.rs` | Utility functions |

### Key Types

- `AffinePoint` / `ExtendedPoint` — curve point representations
- `Fr` (aka `JubJubScalar`) — scalar field
- `Fq` (aka `BlsScalar`) — base field (aliased from dusk-bls12_381)
- Dusk additions: `dhke()`, `GENERATOR`, `GENERATOR_NUMS`

### Features (defaults: alloc, bits)

- `alloc` — allocation support
- `bits` — bit operations on field elements
- `rkyv-impl` — rkyv zero-copy serialization
- `serde` — JSON serialization (includes serde_json, hex)
- `zeroize` — secure memory zeroing

## Conventions

- **no_std by default**: the crate is `no_std`. Do not add `std` dependencies.
- **Dusk submodule scoping**: all Dusk additions go in `dusk` submodules
  (`src/dusk.rs`, `src/fr/dusk.rs`, etc.). Never modify upstream zkcrypto code.
- **No timing side-channels**: do not introduce branches or early returns on
  secret data. Use constant-time operations.
- **Montgomery form**: `Fr` values must be in Montgomery form for public APIs.
  Raw-form values are only for internal arithmetic (e.g. WNAF).
- **Test both configurations**: always run `make test`, which covers
  `--features=zeroize,serde` and `--no-default-features`.

## Changelog

- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change
- Use the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.
- Follow standard markdown formatting: separate headings from surrounding content with blank lines, leave a blank line before and after lists, and never have two headings back-to-back without a blank line between them
