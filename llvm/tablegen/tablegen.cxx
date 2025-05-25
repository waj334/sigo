
#include "tablegen.h"

#include <llvm/Support/SourceMgr.h>
#include <llvm/TableGen/Parser.h>
#include <llvm/TableGen/Record.h>

#include <string>
#include <map>
#include <memory>

using RecordMap = std::map<std::string, std::unique_ptr<llvm::Record>, std::less<>>;
using GlobalMap = std::map<std::string, const llvm::Init *, std::less<>>;

extern "C" {

struct LLVMInit
{
    const llvm::Init* PTR;
};

struct LLVMGlobalMap
{
    GlobalMap* PTR;
};

struct LLVMGlobalMapIterator
{
    const GlobalMap* map;
    int position;
};

struct LLVMRecordList {
    const llvm::Record **records;
    size_t size;
};

struct LLVMRecord
{
    const llvm::Record* PTR;
};

struct LLVMRecordMap
{
    RecordMap* PTR;
};

struct LLVMRecordMapIterator
{
    const RecordMap* map;
    int position;
};

struct LLVMRecordKeeper
{
    llvm::RecordKeeper* PTR;
};

struct LLVMCStringList
{
    std::vector<llvm::StringRef> vec;
};

LLVMRecord* LLVMCreateRecord(const llvm::Record* record);
char* LLVMCreateCString(const std::string& str);
LLVMInit* LLVMCreateInit(const llvm::Init* init);
LLVMGlobalMap* LLVMCreateGlobalMap(const GlobalMap& gm);
LLVMRecordList* LLVMCreateRecordList(const std::vector<const llvm::Record*>& records);
LLVMRecordMap* LLVMCreateRecordMap(const RecordMap& rm);
LLVMCStringList* LLVMCreateCStringList(const std::vector<llvm::StringRef>& vec);

bool LLVMTableGenParseFile(const char* filename,
                            LLVMRecordKeeper* RK,
                            const char **includeDirs,
                            size_t includeDirCount)
{
    // Set include directories.
    std::vector<std::string> dirs;
    for (size_t i = 0; i < includeDirCount; ++i) {
        dirs.emplace_back(includeDirs[i]);
    }

    llvm::SourceMgr srcMgr;
    srcMgr.setIncludeDirs(dirs);

    // Read the file into a buffer.
    auto BufferOrError = llvm::MemoryBuffer::getFile(filename);
    if (!BufferOrError) {
        llvm::errs() << "Could not read file: " << filename << "\n";
        return true; // signal error
    }

    std::unique_ptr<llvm::MemoryBuffer> &Buffer = *BufferOrError;

    // Tell SourceMgr to use this as the main file.
    srcMgr.AddNewSourceBuffer(std::move(Buffer), llvm::SMLoc());

    // Now call the TableGen parser.
    return !llvm::TableGenParseFile(srcMgr, *RK->PTR);
}

char* LLVMCreateCString(const std::string& str)
{
    char *cstr = new char[str.size() + 1];
    std::copy(str.begin(), str.end(), cstr);
    cstr[str.size()] = '\0';
    return cstr;
}

void LLVMDisposeCString(const char* str)
{
    if (str)
    {
        delete[] str;
    }
}

LLVMInit* LLVMCreateInit(const llvm::Init* init)
{
    LLVMInit* I = new LLVMInit{init};
    return I;
}

void LLVMDisposeInit(LLVMInit* I)
{
    if (I)
    {
        delete I;
    }
}

LLVMGlobalMap* LLVMCreateGlobalMap(const GlobalMap& gm)
{
    LLVMGlobalMap* GM = new LLVMGlobalMap();
    GM->PTR = new GlobalMap(gm);
    return GM;
}

void LLVMDisposeGlobalMap(LLVMGlobalMap* GM)
{
    if (GM)
    {
        delete GM->PTR;
        delete GM;
    }
}

LLVMGlobalMapIterator* LLVMGlobalMapBegin(LLVMGlobalMap* GM)
{
    if (!GM || !GM->PTR)
    {
        return nullptr;
    }

    LLVMGlobalMapIterator* IT = new LLVMGlobalMapIterator();
    IT->map = GM->PTR;
    IT->position = -1;

    return IT;
}

void LLVMDisposeGlobalMapIterator(LLVMGlobalMapIterator* IT)
{
    if (IT)
    {
        delete IT;
    }
}

const char* LLVMGlobalMapIteratorKey(LLVMGlobalMapIterator* IT)
{
    const auto it = std::next(IT->map->begin(), IT->position);
    return LLVMCreateCString(it->first);
}

LLVMInit* LLVMGlobalMapIteratorValue(LLVMGlobalMapIterator* IT)
{
    const auto it = std::next(IT->map->begin(), IT->position);
    return LLVMCreateInit(it->second);
}

bool LLVMGlobalMapIteratorNext(LLVMGlobalMapIterator* IT)
{
    IT->position++;
    return IT->position < IT->map->size();
}

LLVMRecordList* LLVMCreateRecordList(const std::vector<const llvm::Record*>& records)
{
    LLVMRecordList* result = new LLVMRecordList();
    result->size = records.size();
    result->records = nullptr;

    if (!records.empty()) {
        result->records = new const llvm::Record*[records.size()];
        std::copy(records.begin(), records.end(), result->records);
    }

    return result;
}

void LLVMDisposeRecordList(LLVMRecordList* list)
{
    delete[] list->records;
    delete list;
}

int LLVMRecordListSize(LLVMRecordList* list)
{
    return list->size;
}

LLVMRecord* LLVMRecordListValue(LLVMRecordList* list, int index)
{
    return LLVMCreateRecord(list->records[index]);
}

LLVMRecord* LLVMCreateRecord(const llvm::Record* record)
{
    LLVMRecord* R = new LLVMRecord{record};
    return R;
}

void LLVMDisposeRecord(LLVMRecord* R)
{
    if (R)
    {
        delete R;
    }
}

bool LLVMRecordIsNull(const LLVMRecord* R)
{
    return R == nullptr || R->PTR == nullptr;
}

const char* LLVMRecordGetName(LLVMRecord* R)
{
    return LLVMCreateCString(R->PTR->getName().str());
}

const char* LLVMRecordGetValueAsString(LLVMRecord* R, const char* name)
{
    return LLVMCreateCString(R->PTR->getValueAsString(name).str());
}

LLVMRecord* LLVMRecordGetValueAsDef(LLVMRecord* R, const char* name)
{
    return LLVMCreateRecord(R->PTR->getValueAsDef(name));
}

LLVMRecord* LLVMRecordGetValueAsOptionalDef(LLVMRecord* R, const char* name)
{
    return LLVMCreateRecord(R->PTR->getValueAsOptionalDef(name));
}

bool LLVMRecordGetValueAsBit(LLVMRecord* R, const char* name)
{
    return R->PTR->getValueAsBit(name);
}

bool LLVMRecordGetValueAsBitOrUnset(LLVMRecord* R, const char* name, bool* unset)
{
    return R->PTR->getValueAsBitOrUnset(name, *unset);
}

int64_t LLVMRecordGetValueAsInt(LLVMRecord* R, const char* name)
{
    return R->PTR->getValueAsInt(name);
}

LLVMRecordList* LLVMRecordGetValueAsListOfDefs(LLVMRecord* R, const char* name)
{
    auto defs = R->PTR->getValueAsListOfDefs(name);
    return LLVMCreateRecordList(defs);
}

LLVMCStringList* LLVMRecordGetValueAsListOfStrings(LLVMRecord* R, const char* name)
{
    const auto values = R->PTR->getValueAsListOfStrings(name);
    return LLVMCreateCStringList(values);
}

void LLVMDisposeRecordMapIterator(LLVMRecordMapIterator* IT)
{
    if (IT)
    {
        delete IT;
    }
}

const char* LLVMRecordMapIteratorKey(LLVMRecordMapIterator* IT)
{
    const auto it = std::next(IT->map->begin(), IT->position);
    return LLVMCreateCString(it->first);
}

LLVMRecord* LLVMRecordMapIteratorValue(LLVMRecordMapIterator* IT)
{
    const auto it = std::next(IT->map->begin(), IT->position);
    return LLVMCreateRecord(it->second.get());
}

bool LLVMRecordMapIteratorNext(LLVMRecordMapIterator* IT)
{
    IT->position++;
    return IT->position < IT->map->size();
}

LLVMRecordMap* LLVMCreateRecordMap(const RecordMap& rm)
{
    LLVMRecordMap* RM = new LLVMRecordMap();
    RM->PTR = new RecordMap();
    for (const auto& [k, v] : rm)
    {
        RM->PTR->emplace(k, std::make_unique<llvm::Record>(*v));
    }
    return RM;
}

void LLVMDisposeRecordMap(LLVMRecordMap* RM)
{
    if (RM)
    {
        delete RM->PTR;
        delete RM;
    }
}

LLVMRecordMapIterator* LLVMRecordMapBegin(LLVMRecordMap* RM)
{
    if (!RM || !RM->PTR)
    {
        return nullptr;
    }

    LLVMRecordMapIterator* IT = new LLVMRecordMapIterator();
    IT->map = RM->PTR;
    IT->position = -1;

    return IT;
}

LLVMRecordKeeper* LLVMCreateRecordKeeper()
{
    LLVMRecordKeeper* RK = new LLVMRecordKeeper();
    RK->PTR = new llvm::RecordKeeper();
    return RK;
}

void LLVMDisposeRecordKeeper(LLVMRecordKeeper* RK)
{
    if (RK)
    {
        delete RK->PTR;
        delete RK;
    }
}

const char* LLVMRecordKeeperGetInputFilename(LLVMRecordKeeper* RK)
{
    return LLVMCreateCString(RK->PTR->getInputFilename());
}

LLVMRecordMap* LLVMRecordKeeperGetClasses(LLVMRecordKeeper* RK)
{
    return LLVMCreateRecordMap(RK->PTR->getClasses());
}

LLVMRecordMap* LLVMRecordKeeperGetDefs(LLVMRecordKeeper* RK)
{
    return LLVMCreateRecordMap(RK->PTR->getDefs());
}

LLVMGlobalMap* LLVMRecordKeeperGetGlobals(LLVMRecordKeeper* RK)
{
    return LLVMCreateGlobalMap(RK->PTR->getGlobals());
}

LLVMRecordList* LLVMRecordKeeperGetAllDerivedDefinitions(LLVMRecordKeeper *RK, const char* className)
{
    auto defs = RK->PTR->getAllDerivedDefinitions(className);
    return LLVMCreateRecordList(defs);
}

LLVMCStringList* LLVMCreateCStringList(const std::vector<llvm::StringRef>& vec)
{
    LLVMCStringList* SL = new LLVMCStringList();
    SL->vec = vec;
    return SL;
}

void LLVMDisposeCStringList(LLVMCStringList* SL)
{
    if (SL)
    {
        delete SL;
    }
}

int LLVMCStringListSize(LLVMCStringList* SL)
{
    return static_cast<int>(SL->vec.size());
}

const char* LLVMCStringListValue(LLVMCStringList* SL, int index)
{
    return LLVMCreateCString(SL->vec[index].str());
}

}