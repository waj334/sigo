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

struct LLVMRecord
{
    llvm::Record* PTR;
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

LLVMRecord* LLVMCreateRecord(const std::unique_ptr<llvm::Record>& record)
{
    LLVMRecord* R = new LLVMRecord();
    R->PTR = record.get();
    return R;
}

void LLVMDisposeRecord(LLVMRecord* R)
{
    if (R)
    {
        delete R;
    }
}

const char* LLVMRecordGetName(LLVMRecord* R)
{
    return LLVMCreateCString(R->PTR->getName().str());
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
    return LLVMCreateRecord(it->second);
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

}