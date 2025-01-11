#ifndef GO_ATTR_TD
#define GO_ATTR_TD

include "Go/IR/GoDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

class GoDialect_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<Go_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def Go_TypeMetadataAttr : GoDialect_Attr<"TypeMetadata", "go_type_metadata"> {
    let summary = "An attribute representing a Go type";
    let description = [{
        This attribute holds all of the information describing a specific Go
        data type.
    }];

    let parameters = (ins
        "TypeAttr":$type,
        "DictionaryAttr":$metadata
    );
    let assemblyFormat = [{
        `<` params `>`
    }];
}

//===----------------------------------------------------------------------===//
// TypeMetadataEntryAttr
//===----------------------------------------------------------------------===//

def TypeMetadataEntryAttr : GoDialect_Attr<"TypeMetadataEntry", "type_metadata_entry"> {
    let summary = "An attribute representing a Go type";
    let description = [{
        This attribute holds all of the information describing a specific Go
        data type.
    }];

    let parameters = (ins
        "TypeAttr":$type,
        "DictionaryAttr":$metadata
    );
    let assemblyFormat = [{
        `<` params `>`
    }];
}

//===----------------------------------------------------------------------===//
// TypeMetadataEntryArrayAttr
//===----------------------------------------------------------------------===//

def TypeMetadataEntryArrayAttr : ArrayOfAttr<Go_Dialect,
                                     "TypeMetadataEntryArray",
                                     "type_metadata_entry_array",
                                     TypeMetadataEntryAttr.cppClassName> {}

//===----------------------------------------------------------------------===//
// TypeMetadataDictionaryAttr
//===----------------------------------------------------------------------===//

def TypeMetadataDictionaryAttr : GoDialect_Attr<"TypeMetadataDictionary", "type_metadata_dictionary"> {
    let parameters = (ins
        "TypeMetadataEntryArrayAttr":$entries
    );
    let assemblyFormat = "`<` params `>`";
}

//===----------------------------------------------------------------------===//
// ComplexAttr
//===----------------------------------------------------------------------===//

def ComplexNumberAttr : GoDialect_Attr<"ComplexNumber", "complex_number"> {
    let parameters = (ins
        "FloatAttr":$real,
        "FloatAttr":$imag
    );
    let assemblyFormat = "`<` params `>`";
}

//===----------------------------------------------------------------------===//
// RuntimeTypeEntryAttr
//===----------------------------------------------------------------------===//

def RuntimeTypeEntryAttr : GoDialect_Attr<"RuntimeTypeEntry", "runtime_type_entry", [
            DeclareAttrInterfaceMethods<DataLayoutEntryInterface, ["getKey", "getValue"]>
        ]> {
    let parameters = (ins
        "::mlir::TypeAttr":$primitiveType,
        "::mlir::TypeAttr":$runtimeType
    );
    let extraClassDefinition = [{
        ::mlir::DataLayoutEntryKey RuntimeTypeEntryAttr::getKey() const { return ::mlir::DataLayoutEntryKey(this->getPrimitiveType().getValue()); }
        ::mlir::Attribute RuntimeTypeEntryAttr::getValue() const { return this->getRuntimeType(); }
    }];
    let assemblyFormat = "`<` params `>`";
}

//===----------------------------------------------------------------------===//
// AsmConstraintDirection
//===----------------------------------------------------------------------===//

def AsmConstraintDirectionAttr : GoDialect_Attr<"AsmConstraintDirection", "asm_constraint_dir"> {
    let parameters = (ins
        OptionalParameter<"UnitAttr">:$input,
        OptionalParameter<"UnitAttr">:$output
    );
    let assemblyFormat = "`<` (`in` $input^)? (`out` $output^)? `>`";
}

//===----------------------------------------------------------------------===//
// AsmConstraint
//===----------------------------------------------------------------------===//

def AsmConstraintAttr : GoDialect_Attr<"AsmConstraint", "asm_constraint"> {
    let parameters = (ins
        "StringAttr":$registerClass,
        "AsmConstraintDirectionAttr":$direction,
        OptionalParameter<"StringAttr">:$alias,
        OptionalParameter<"IntegerAttr">:$operand_index,
        OptionalParameter<"UnitAttr">:$reserve
    );
    let assemblyFormat = "`<` $registerClass ` ` $direction (` ` $alias^)? (` ` $operand_index^)? (` ` $reserve^ `reserve`)? `>`";
}


#endif // GO_ATTR_TD
