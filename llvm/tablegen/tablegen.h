#ifndef C_LLVM_TABLEGEN_H
#define C_LLVM_TABLEGEN_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void LLVMDisposeCString(const char* str);

typedef struct LLVMInit LLVMInit;
void LLVMDisposeInit(LLVMInit* I);

typedef struct LLVMGlobalMapIterator LLVMGlobalMapIterator;
void LLVMDisposeGlobalMapIterator(LLVMGlobalMapIterator* IT);
const char* LLVMGlobalMapIteratorKey(LLVMGlobalMapIterator* IT);
LLVMInit* LLVMGlobalMapIteratorValue(LLVMGlobalMapIterator* IT);
bool LLVMGlobalMapIteratorNext(LLVMGlobalMapIterator* IT);

typedef struct LLVMGlobalMap LLVMGlobalMap;
void LLVMDisposeGlobalMap(LLVMGlobalMap* GM);
LLVMGlobalMapIterator* LLVMGlobalMapBegin(LLVMGlobalMap* GM);

typedef struct LLVMRecord LLVMRecord;
void LLVMDisposeRecord(LLVMRecord* R);
const char* LLVMRecordGetName(LLVMRecord* R);

typedef struct LLVMRecordMapIterator LLVMRecordMapIterator;
void LLVMDisposeRecordMapIterator(LLVMRecordMapIterator* IT);
const char* LLVMRecordMapIteratorKey(LLVMRecordMapIterator* IT);
LLVMRecord* LLVMRecordMapIteratorValue(LLVMRecordMapIterator* IT);
bool LLVMRecordMapIteratorNext(LLVMRecordMapIterator* IT);

typedef struct LLVMRecordMap LLVMRecordMap;
void LLVMDisposeRecordMap(LLVMRecordMap* RM);
LLVMRecordMapIterator* LLVMRecordMapBegin(LLVMRecordMap* RM);

typedef struct LLVMRecordKeeper LLVMRecordKeeper;
LLVMRecordKeeper* LLVMCreateRecordKeeper();
void LLVMDisposeRecordKeeper(LLVMRecordKeeper* RK);
const char* LLVMRecordKeeperGetInputFilename(LLVMRecordKeeper* RK);
LLVMRecordMap* LLVMRecordKeeperGetClasses(LLVMRecordKeeper* RK);
LLVMRecordMap* LLVMRecordKeeperGetDefs(LLVMRecordKeeper* RK);
LLVMGlobalMap* LLVMRecordKeeperGetGlobals(LLVMRecordKeeper* RK);

bool LLVMTableGenParseFile(const char* filename,
                            LLVMRecordKeeper* RK,
                            const char **includeDirs,
                            size_t includeDirCount);

#ifdef __cplusplus
}
#endif

#endif // C_LLVM_TABLEGEN_H