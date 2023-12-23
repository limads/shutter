// Original version: https://docs.rs/tract-onnx/0.20.21/src/tract_onnx/prost/onnx.rs.html#7


use prost::Message;


#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelProto {

    #[prost(int64, tag="1")]
    pub ir_version: i64,

    #[prost(message, repeated, tag="8")]
    pub opset_import: ::prost::alloc::vec::Vec<OperatorSetIdProto>,

    #[prost(string, tag="2")]
    pub producer_name: ::prost::alloc::string::String,

    #[prost(string, tag="3")]
    pub producer_version: ::prost::alloc::string::String,

    #[prost(string, tag="4")]
    pub domain: ::prost::alloc::string::String,

    #[prost(int64, tag="5")]
    pub model_version: i64,

    #[prost(string, tag="6")]
    pub doc_string: ::prost::alloc::string::String,

    #[prost(message, optional, tag="7")]
    pub graph: ::core::option::Option<GraphProto>,

    #[prost(message, repeated, tag="14")]
    pub metadata_props: ::prost::alloc::vec::Vec<StringStringEntryProto>,

    #[prost(message, repeated, tag="20")]
    pub training_info: ::prost::alloc::vec::Vec<TrainingInfoProto>,

    #[prost(message, repeated, tag="25")]
    pub functions: ::prost::alloc::vec::Vec<FunctionProto>,
}


#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphProto {

    #[prost(message, repeated, tag="1")]
    pub node: ::prost::alloc::vec::Vec<NodeProto>,

    #[prost(string, tag="2")]
    pub name: ::prost::alloc::string::String,

    #[prost(message, repeated, tag="5")]
    pub initializer: ::prost::alloc::vec::Vec<TensorProto>,

    #[prost(message, repeated, tag="15")]
    pub sparse_initializer: ::prost::alloc::vec::Vec<SparseTensorProto>,

    #[prost(string, tag="10")]
    pub doc_string: ::prost::alloc::string::String,

    #[prost(message, repeated, tag="11")]
    pub input: ::prost::alloc::vec::Vec<ValueInfoProto>,

    #[prost(message, repeated, tag="12")]
    pub output: ::prost::alloc::vec::Vec<ValueInfoProto>,

    #[prost(message, repeated, tag="13")]
    pub value_info: ::prost::alloc::vec::Vec<ValueInfoProto>,

    #[prost(message, repeated, tag="14")]
    pub quantization_annotation: ::prost::alloc::vec::Vec<TensorAnnotation>,

}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttributeProto {

    #[prost(string, tag="1")]
    pub name: ::prost::alloc::string::String,

    #[prost(string, tag="21")]
    pub ref_attr_name: ::prost::alloc::string::String,

    #[prost(string, tag="13")]
    pub doc_string: ::prost::alloc::string::String,

    #[prost(enumeration="attribute_proto::AttributeType", tag="20")]
    pub r#type: i32,

    #[prost(float, tag="2")]
    pub f: f32,

    #[prost(int64, tag="3")]
    pub i: i64,

    #[prost(bytes="vec", tag="4")]
    pub s: ::prost::alloc::vec::Vec<u8>,

    #[prost(message, optional, tag="5")]
    pub t: ::core::option::Option<TensorProto>,

    #[prost(message, optional, tag="6")]
    pub g: ::core::option::Option<GraphProto>,

    #[prost(message, optional, tag="22")]
    pub sparse_tensor: ::core::option::Option<SparseTensorProto>,

    #[prost(float, repeated, tag="7")]
    pub floats: ::prost::alloc::vec::Vec<f32>,

    #[prost(int64, repeated, tag="8")]
    pub ints: ::prost::alloc::vec::Vec<i64>,

    #[prost(bytes="vec", repeated, tag="9")]
    pub strings: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,

    #[prost(message, repeated, tag="10")]
    pub tensors: ::prost::alloc::vec::Vec<TensorProto>,

    #[prost(message, repeated, tag="11")]
    pub graphs: ::prost::alloc::vec::Vec<GraphProto>,

    #[prost(message, repeated, tag="23")]
    pub sparse_tensors: ::prost::alloc::vec::Vec<SparseTensorProto>,

    #[prost(message, repeated, tag="15")]
    pub type_protos: ::prost::alloc::vec::Vec<TypeProto>,
}

pub mod attribute_proto {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum AttributeType {
        Undefined = 0,
        Float = 1,
        Int = 2,
        String = 3,
        Tensor = 4,
        Graph = 5,
        SparseTensor = 11,
        TypeProto = 13,
        Floats = 6,
        Ints = 7,
        Strings = 8,
        Tensors = 9,
        Graphs = 10,
        SparseTensors = 12,
        TypeProtos = 14,
    }

    impl AttributeType {

        pub fn as_str_name(&self) -> &'static str {
            match self {
                AttributeType::Undefined => "UNDEFINED",
                AttributeType::Float => "FLOAT",
                AttributeType::Int => "INT",
                AttributeType::String => "STRING",
                AttributeType::Tensor => "TENSOR",
                AttributeType::Graph => "GRAPH",
                AttributeType::SparseTensor => "SPARSE_TENSOR",
                AttributeType::TypeProto => "TYPE_PROTO",
                AttributeType::Floats => "FLOATS",
                AttributeType::Ints => "INTS",
                AttributeType::Strings => "STRINGS",
                AttributeType::Tensors => "TENSORS",
                AttributeType::Graphs => "GRAPHS",
                AttributeType::SparseTensors => "SPARSE_TENSORS",
                AttributeType::TypeProtos => "TYPE_PROTOS",
            }
        }
    }

}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ValueInfoProto {

    #[prost(string, tag="1")]
    pub name: ::prost::alloc::string::String,

    #[prost(message, optional, tag="2")]
    pub r#type: ::core::option::Option<TypeProto>,

    #[prost(string, tag="3")]
    pub doc_string: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeProto {

    #[prost(string, repeated, tag="1")]
    pub input: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,

    #[prost(string, repeated, tag="2")]
    pub output: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,

    #[prost(string, tag="3")]
    pub name: ::prost::alloc::string::String,

    #[prost(string, tag="4")]
    pub op_type: ::prost::alloc::string::String,

    #[prost(string, tag="7")]
    pub domain: ::prost::alloc::string::String,

    #[prost(message, repeated, tag="5")]
    pub attribute: ::prost::alloc::vec::Vec<AttributeProto>,

    #[prost(string, tag="6")]
    pub doc_string: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag="1")]
    pub key: ::prost::alloc::string::String,
    #[prost(string, tag="2")]
    pub value: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorAnnotation {
    #[prost(string, optional, tag="1")]
    pub tensor_name: ::core::option::Option<::prost::alloc::string::String>,

    #[prost(message, repeated, tag="2")]
    pub quant_parameter_tensor_names: ::prost::alloc::vec::Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TrainingInfoProto {

    #[prost(message, optional, tag="1")]
    pub initialization: ::core::option::Option<GraphProto>,

    #[prost(message, optional, tag="2")]
    pub algorithm: ::core::option::Option<GraphProto>,

    #[prost(message, repeated, tag="3")]
    pub initialization_binding: ::prost::alloc::vec::Vec<StringStringEntryProto>,

    #[prost(message, repeated, tag="4")]
    pub update_binding: ::prost::alloc::vec::Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProto {

    #[prost(int64, repeated, tag="1")]
    pub dims: ::prost::alloc::vec::Vec<i64>,

    #[prost(enumeration="tensor_proto::DataType", tag="2")]
    pub data_type: i32,
    #[prost(message, optional, tag="3")]
    pub segment: ::core::option::Option<tensor_proto::Segment>,

    #[prost(float, repeated, tag="4")]
    pub float_data: ::prost::alloc::vec::Vec<f32>,

    #[prost(int32, repeated, tag="5")]
    pub int32_data: ::prost::alloc::vec::Vec<i32>,

    #[prost(bytes="vec", repeated, tag="6")]
    pub string_data: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,

    #[prost(int64, repeated, tag="7")]
    pub int64_data: ::prost::alloc::vec::Vec<i64>,

    #[prost(string, tag="8")]
    pub name: ::prost::alloc::string::String,

    #[prost(string, tag="12")]
    pub doc_string: ::prost::alloc::string::String,

    #[prost(bytes="vec", tag="9")]
    pub raw_data: ::prost::alloc::vec::Vec<u8>,

    #[prost(double, repeated, tag="10")]
    pub double_data: ::prost::alloc::vec::Vec<f64>,

    #[prost(uint64, repeated, tag="11")]
    pub uint64_data: ::prost::alloc::vec::Vec<u64>,

    #[prost(enumeration="tensor_proto::DataLocation", optional, tag="14")]
    pub data_location: ::core::option::Option<i32>,

    #[prost(message, repeated, tag="13")]
    pub external_data: ::prost::alloc::vec::Vec<StringStringEntryProto>,
}

pub mod tensor_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Segment {
        #[prost(int64, tag="1")]
        pub begin: i64,
        #[prost(int64, tag="2")]
        pub end: i64,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DataType {
        Undefined = 0,

        Float = 1,

        Uint8 = 2,

        Int8 = 3,

        Uint16 = 4,

        Int16 = 5,

        Int32 = 6,

        Int64 = 7,

        String = 8,

        Bool = 9,

        Float16 = 10,
        Double = 11,
        Uint32 = 12,
        Uint64 = 13,

        Complex64 = 14,

        Complex128 = 15,

        Bfloat16 = 16,
    }

    impl DataType {

        pub fn as_str_name(&self) -> &'static str {
            match self {
                DataType::Undefined => "UNDEFINED",
                DataType::Float => "FLOAT",
                DataType::Uint8 => "UINT8",
                DataType::Int8 => "INT8",
                DataType::Uint16 => "UINT16",
                DataType::Int16 => "INT16",
                DataType::Int32 => "INT32",
                DataType::Int64 => "INT64",
                DataType::String => "STRING",
                DataType::Bool => "BOOL",
                DataType::Float16 => "FLOAT16",
                DataType::Double => "DOUBLE",
                DataType::Uint32 => "UINT32",
                DataType::Uint64 => "UINT64",
                DataType::Complex64 => "COMPLEX64",
                DataType::Complex128 => "COMPLEX128",
                DataType::Bfloat16 => "BFLOAT16",
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DataLocation {
        Default = 0,
        External = 1,
    }
    impl DataLocation {
        pub fn as_str_name(&self) -> &'static str {
            match self {
                DataLocation::Default => "DEFAULT",
                DataLocation::External => "EXTERNAL",
            }
        }
    }

}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SparseTensorProto {

    #[prost(message, optional, tag="1")]
    pub values: ::core::option::Option<TensorProto>,

    #[prost(message, optional, tag="2")]
    pub indices: ::core::option::Option<TensorProto>,

    #[prost(int64, repeated, tag="3")]
    pub dims: ::prost::alloc::vec::Vec<i64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag="1")]
    pub dim: ::prost::alloc::vec::Vec<tensor_shape_proto::Dimension>,
}

pub mod tensor_shape_proto {
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Dimension {

    #[prost(string, tag="3")]
    pub denotation: ::prost::alloc::string::String,
    #[prost(oneof="dimension::Value", tags="1, 2")]
    pub value: ::core::option::Option<dimension::Value>,
}

    pub mod dimension {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Value {
            #[prost(int64, tag="1")]
            DimValue(i64),
            #[prost(string, tag="2")]
            DimParam(::prost::alloc::string::String),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeProto {

    #[prost(string, tag="6")]
    pub denotation: ::prost::alloc::string::String,
    #[prost(oneof="type_proto::Value", tags="1")]
    pub value: ::core::option::Option<type_proto::Value>,
}

pub mod type_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Tensor {

        #[prost(enumeration="super::tensor_proto::DataType", tag="1")]
        pub elem_type: i32,
        #[prost(message, optional, tag="2")]
        pub shape: ::core::option::Option<super::TensorShapeProto>,
    }

    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {

        #[prost(message, tag="1")]
        TensorType(Tensor),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OperatorSetIdProto {

    #[prost(string, tag="1")]
    pub domain: ::prost::alloc::string::String,

    #[prost(int64, tag="2")]
    pub version: i64,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FunctionProto {

    #[prost(string, optional, tag="1")]
    pub name: ::core::option::Option<::prost::alloc::string::String>,

    #[prost(string, repeated, tag="4")]
    pub input: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(string, repeated, tag="5")]
    pub output: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,

    #[prost(string, repeated, tag="6")]
    pub attribute: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,

    #[prost(message, repeated, tag="7")]
    pub node: ::prost::alloc::vec::Vec<NodeProto>,

    #[prost(string, optional, tag="8")]
    pub doc_string: ::core::option::Option<::prost::alloc::string::String>,

    #[prost(message, repeated, tag="9")]
    pub opset_import: ::prost::alloc::vec::Vec<OperatorSetIdProto>,

    #[prost(string, optional, tag="10")]
    pub domain: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum Version {

    StartVersion = 0,

    IrVersion20171010 = 1,

    IrVersion20171030 = 2,

    IrVersion2017113 = 3,

    IrVersion2019122 = 4,

    IrVersion2019318 = 5,

    IrVersion2019919 = 6,

    IrVersion202058 = 7,

    IrVersion = 8,
}
impl Version {

    pub fn as_str_name(&self) -> &'static str {
        match self {
            Version::StartVersion => "_START_VERSION",
            Version::IrVersion20171010 => "IR_VERSION_2017_10_10",
            Version::IrVersion20171030 => "IR_VERSION_2017_10_30",
            Version::IrVersion2017113 => "IR_VERSION_2017_11_3",
            Version::IrVersion2019122 => "IR_VERSION_2019_1_22",
            Version::IrVersion2019318 => "IR_VERSION_2019_3_18",
            Version::IrVersion2019919 => "IR_VERSION_2019_9_19",
            Version::IrVersion202058 => "IR_VERSION_2020_5_8",
            Version::IrVersion => "IR_VERSION",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum OperatorStatus {
    Experimental = 0,
    Stable = 1,
}
impl OperatorStatus {

    pub fn as_str_name(&self) -> &'static str {
        match self {
            OperatorStatus::Experimental => "EXPERIMENTAL",
            OperatorStatus::Stable => "STABLE",
        }
    }
}

// cargo test --lib -- onnx_read --nocapture
#[test]
fn onnx_read() {
    let bytes = std::fs::read("/home/diego/Downloads/model.onnx").unwrap();
    let m = ModelProto::decode(&bytes[..]).unwrap();
    let graph = &m.graph.unwrap();
    for node in &graph.node {
        println!("{}", node.name);
        println!("In {:?}", node.input);
        println!("Out {:?}", node.output);
        println!("Op {}", node.op_type);

        // Conv
        // Transpose (perm (ints) carry the new indices of the dimensions)
        // Reshape
        // Selu
        // MaxPool

        // graph.input[0].name
        // graph.output[0].name

        for attr in &node.attribute {
            println!("{}", attr.name);
        }
        println!("------");
    }
}

fn get_max_pool(node : &NodeProto) {
    for attr in &node.attribute {
        match &attr.name[..] {
            "strides" => {

            },
            "kernel_shape" => {

            },
            _ => { }
        }
    }
}

fn get_conv(node : &NodeProto) {
    for attr in &node.attribute {
        match &attr.name[..] {
            "strides" => {

            },
            "kernel_shape" => {

            },
            _ => { }
        }
    }
}

pub struct Step {
    input : Vec<f32>,
    output: Vec<f32>
}

// Reshape: Receives data tensor as LHS; Dimension tensor as RHS.
// Conv: Receives data as LHS; Receives kernel as RHS.

pub enum Op {

}

// To implement transposed convolution:
// (1) Pad original input with zeros up to the desired size (and add zeroes between
// the pixels of the original inuput when stride > 1)
// (2) Apply a convolution.
