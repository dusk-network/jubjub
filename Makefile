CI_FEATURES := alloc,bits,rkyv-impl,serde,zeroize,rkyv/size_32
TEST_FEATURES := zeroize,serde
BLST_BACKEND := bls-backend-blst
DUSK_BACKEND := bls-backend-dusk

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: test-blst test-dusk ## Run tests (std + no_std, both BLS backends)

test-blst: ## Run tests with the blst BLS backend
	@cargo test --features=$(TEST_FEATURES),$(BLST_BACKEND)
	@cargo test --no-default-features --features=$(BLST_BACKEND)

test-dusk: ## Run tests with the Dusk BLS backend
	@cargo test --features=$(TEST_FEATURES),$(DUSK_BACKEND)
	@cargo test --no-default-features --features=$(DUSK_BACKEND)

clippy: clippy-blst clippy-dusk ## Run clippy with both BLS backends

clippy-blst: ## Run clippy with the blst BLS backend
	@cargo clippy --features=$(CI_FEATURES),$(BLST_BACKEND) -- -D warnings

clippy-dusk: ## Run clippy with the Dusk BLS backend
	@cargo clippy --features=$(CI_FEATURES),$(DUSK_BACKEND) -- -D warnings

fmt: ## Format code
	@cargo +nightly fmt --all

check: check-blst check-dusk ## Type-check with both BLS backends

check-blst: ## Type-check with the blst BLS backend
	@cargo check --features=$(CI_FEATURES),$(BLST_BACKEND)

check-dusk: ## Type-check with the Dusk BLS backend
	@cargo check --features=$(CI_FEATURES),$(DUSK_BACKEND)

doc: doc-blst doc-dusk ## Generate docs with both BLS backends

doc-blst: ## Generate docs with the blst BLS backend
	@cargo doc --no-deps --features=$(BLST_BACKEND)

doc-dusk: ## Generate docs with the Dusk BLS backend
	@cargo doc --no-deps --features=$(DUSK_BACKEND)

clean: ## Clean build artifacts
	@cargo clean

no-std: no-std-blst no-std-dusk ## Verify no_std + WASM compatibility with both BLS backends

no-std-blst: ## Verify no_std + WASM compatibility with the blst BLS backend
	@rustup target add wasm32-unknown-unknown 2>/dev/null || true
	@cargo build --release --no-default-features --features serde,$(BLST_BACKEND) --target wasm32-unknown-unknown

no-std-dusk: ## Verify no_std + WASM compatibility with the Dusk BLS backend
	@rustup target add wasm32-unknown-unknown 2>/dev/null || true
	@cargo build --release --no-default-features --features serde,$(DUSK_BACKEND) --target wasm32-unknown-unknown

.PHONY: help test test-blst test-dusk clippy clippy-blst clippy-dusk fmt check check-blst check-dusk doc doc-blst doc-dusk clean no-std no-std-blst no-std-dusk
