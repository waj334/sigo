%module clang

%include inttypes.i
%include typemaps.i

%header %{
    #include "clang-c/BuildSystem.h"
    #include "clang-c/CXCompilationDatabase.h"
    #include "clang-c/CXDiagnostic.h"
    #include "clang-c/CXErrorCode.h"
    #include "clang-c/CXFile.h"
    #include "clang-c/CXSourceLocation.h"
    #include "clang-c/CXString.h"
    #include "clang-c/Documentation.h"
    #include "clang-c/ExternC.h"
    #include "clang-c/FatalErrorHandler.h"
    #include "clang-c/Index.h"
    #include "clang-c/module.modulemap"
    #include "clang-c/Platform.h"
    #include "clang-c/Rewrite.h"
%}



%include "clang-c/ExternC.h"
%include "clang-c/Platform.h"
%include "clang-c/BuildSystem.h"
%include "clang-c/CXCompilationDatabase.h"
%include "clang-c/CXDiagnostic.h"
%include "clang-c/CXErrorCode.h"
%include "clang-c/CXFile.h"
%include "clang-c/CXSourceLocation.h"
%include "clang-c/CXString.h"
%include "clang-c/Documentation.h"
%include "clang-c/FatalErrorHandler.h"
%include "clang-c/Index.h"
%include "clang-c/module.modulemap"
%include "clang-c/Rewrite.h"