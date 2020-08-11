#[derive(Debug)]
/// Standard error for the interface
pub enum Error {
    /// Cryptographic bottom
    Generic,
    /// Resource not ready
    NotReady,
    /// The transaction needs to be prepared before it can be stored
    TransactionNotPrepared,
    /// Failed to create the fee output
    FeeOutput,
    /// Invalid compressed point provided
    InvalidPoint,
    /// Invalid parameters provided to the function
    InvalidParameters,
    /// Maximum number of notes per transaction exceeded
    MaximumNotes,
    /// The queried information was not found
    NotFound,
    /// Attempt to double spend
    DoubleSpending,
}

impl Error {
    /// Return a generic error from any type. Represents a cryptographic bottom
    pub fn generic<T>(_e: T) -> Error {
        Error::Generic
    }
}