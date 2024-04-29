#ifndef GO_ATTR_TD
#define GO_ATTR_TD

include "Go/IR/GoDialect.td"
include "mlir/IR/AttrTypeBase.td"

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

#endif // GO_ATTR_TD
