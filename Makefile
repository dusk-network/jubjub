help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests (std + no_std)
	@cargo test --features=zeroize,serde
	@cargo test --no-default-features
	@cargo test --all-features --features=rkyv/size_32,rkyv/validation
	@cargo test --lib --test archive --features=rkyv-validation,rkyv/size_32,rkyv/archive_be

clippy: ## Run clippy
	@cargo clippy --all-features --features=rkyv/size_32 -- -D warnings

fmt: ## Format code
	@cargo +nightly fmt --all

check: ## Type-check
	@cargo check --all-features --features=rkyv/size_32

doc: ## Generate docs
	@cargo doc --no-deps

clean: ## Clean build artifacts
	@cargo clean

no-std: ## Verify no_std + WASM compatibility
	@rustup target add wasm32-unknown-unknown 2>/dev/null || true
	@cargo build --release --no-default-features --features serde --target wasm32-unknown-unknown

.PHONY: help test clippy fmt check doc clean no-std
