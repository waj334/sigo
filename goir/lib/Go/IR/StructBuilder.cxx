#include "Go/IR/StructBuilder.h"

#include "Go/IR/GoOps.h"

namespace mlir::go
{
StructBuilder::StructBuilder(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type T)
  : m_builder(builder)
  , m_structT(T)
  , m_loc(loc)
{
  // Create a zero value of the struct.
  m_currentValue = m_builder.create<ZeroOp>(this->m_loc, baseType(this->m_structT));
}

StructBuilder::StructBuilder(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value)
  : m_builder(builder)
  , m_structT(baseType(value.getType()))
  , m_currentValue(value)
  , m_loc(loc)
{
  // Does nothing.
}

void StructBuilder::Insert(uint64_t index, mlir::Value value)
{
  auto T = mlir::go::cast<LLVM::LLVMStructType>(m_structT);
  m_currentValue = this->m_builder.create<InsertOp>(this->m_loc, T, value, index, m_currentValue);
}

mlir::Value StructBuilder::Value() const
{
  if (auto namedType = mlir::dyn_cast<NamedType>(this->m_structT))
  {
    // Bitcast to the named type
  }
  return this->m_currentValue;
}
} // namespace mlir::go
