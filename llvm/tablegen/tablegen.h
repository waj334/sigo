#ifndef C_LLVM_TABLEGEN_H
#define C_LLVM_TABLEGEN_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLVMInit LLVMInit;
typedef struct LLVMGlobalMapIterator LLVMGlobalMapIterator;
typedef struct LLVMGlobalMap LLVMGlobalMap;
typedef struct LLVMRecord LLVMRecord;
typedef struct LLVMRecordList LLVMRecordList;
typedef struct LLVMRecordMapIterator LLVMRecordMapIterator;
typedef struct LLVMRecordMap LLVMRecordMap;
typedef struct LLVMRecordKeeper LLVMRecordKeeper;

void LLVMDisposeCString(const char* str);

void LLVMDisposeInit(LLVMInit* I);

void LLVMDisposeGlobalMapIterator(LLVMGlobalMapIterator* IT);
const char* LLVMGlobalMapIteratorKey(LLVMGlobalMapIterator* IT);
LLVMInit* LLVMGlobalMapIteratorValue(LLVMGlobalMapIterator* IT);
bool LLVMGlobalMapIteratorNext(LLVMGlobalMapIterator* IT);

void LLVMDisposeGlobalMap(LLVMGlobalMap* GM);
LLVMGlobalMapIterator* LLVMGlobalMapBegin(LLVMGlobalMap* GM);

void LLVMDisposeRecord(LLVMRecord* R);
bool LLVMRecordIsNull(const LLVMRecord* R);
const char* LLVMRecordGetName(LLVMRecord* R);
const char* LLVMRecordGetValueAsString(LLVMRecord* R, const char* name);
LLVMRecord* LLVMRecordGetValueAsDef(LLVMRecord* R, const char* name);
LLVMRecord* LLVMRecordGetValueAsOptionalDef(LLVMRecord* R, const char* name);
bool LLVMRecordGetValueAsBit(LLVMRecord* R, const char* name);
bool LLVMRecordGetValueAsBitOrUnset(LLVMRecord* R, const char* name, bool* unset);
int64_t LLVMRecordGetValueAsInt(LLVMRecord* R, const char* name);
LLVMRecordList* LLVMRecordGetValueAsListOfDefs(LLVMRecord* R, const char* name);

void LLVMDisposeRecordList(LLVMRecordList* list);
int LLVMRecordListSize(LLVMRecordList* list);
LLVMRecord* LLVMRecordListValue(LLVMRecordList* list, int index);

void LLVMDisposeRecordMapIterator(LLVMRecordMapIterator* IT);
const char* LLVMRecordMapIteratorKey(LLVMRecordMapIterator* IT);
LLVMRecord* LLVMRecordMapIteratorValue(LLVMRecordMapIterator* IT);
bool LLVMRecordMapIteratorNext(LLVMRecordMapIterator* IT);

void LLVMDisposeRecordMap(LLVMRecordMap* RM);
LLVMRecordMapIterator* LLVMRecordMapBegin(LLVMRecordMap* RM);

LLVMRecordKeeper* LLVMCreateRecordKeeper();
void LLVMDisposeRecordKeeper(LLVMRecordKeeper* RK);
const char* LLVMRecordKeeperGetInputFilename(LLVMRecordKeeper* RK);
LLVMRecordMap* LLVMRecordKeeperGetClasses(LLVMRecordKeeper* RK);
LLVMRecordMap* LLVMRecordKeeperGetDefs(LLVMRecordKeeper* RK);
LLVMGlobalMap* LLVMRecordKeeperGetGlobals(LLVMRecordKeeper* RK);
LLVMRecordList* LLVMRecordKeeperGetAllDerivedDefinitions(LLVMRecordKeeper *RK, const char *className);

bool LLVMTableGenParseFile(const char* filename,
                            LLVMRecordKeeper* RK,
                            const char **includeDirs,
                            size_t includeDirCount);

#ifdef __cplusplus
}
#endif

#endif // C_LLVM_TABLEGEN_H