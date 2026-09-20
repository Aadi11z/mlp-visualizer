use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkConfigError {
    TooFewLayers,
    TooManyLayers {
        actual: usize,
        maximum: usize,
    },
    ZeroWidth {
        layer: usize,
    },
    LayerTooWide {
        layer: usize,
        actual: usize,
        maximum: usize,
    },
    ActivationCountMismatch {
        expected: usize,
        actual: usize,
    },
    ParameterCountOverflow,
    TooManyParameters {
        actual: usize,
        maximum: usize,
    },
}

impl fmt::Display for NetworkConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewLayers => write!(f, "a network must have input and output layers"),
            Self::TooManyLayers { actual, maximum } => {
                write!(f, "network has {actual} layers; maximum is {maximum}")
            }
            Self::ZeroWidth { layer } => write!(f, "layer {layer} must have at least one neuron"),
            Self::LayerTooWide {
                layer,
                actual,
                maximum,
            } => {
                write!(f, "layer {layer} has width {actual}; maximum is {maximum}")
            }
            Self::ActivationCountMismatch { expected, actual } => {
                write!(f, "expected {expected} activations, received {actual}")
            }
            Self::ParameterCountOverflow => write!(f, "network parameter count overflowed"),
            Self::TooManyParameters { actual, maximum } => {
                write!(f, "network has {actual} parameters; maximum is {maximum}")
            }
        }
    }
}

impl std::error::Error for NetworkConfigError {}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    Config(NetworkConfigError),
    InvalidValue(String),
    ShapeMismatch(String),
    UnsupportedConfiguration(String),
    ResourceLimit(String),
    NumericalFailure(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => write!(f, "{error}"),
            Self::InvalidValue(message)
            | Self::ShapeMismatch(message)
            | Self::UnsupportedConfiguration(message)
            | Self::ResourceLimit(message)
            | Self::NumericalFailure(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<NetworkConfigError> for EngineError {
    fn from(error: NetworkConfigError) -> Self {
        Self::Config(error)
    }
}
