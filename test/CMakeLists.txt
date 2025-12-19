find_program(FILECHECK_EXE
  NAMES FileCheck
  HINTS ${LLVM_TOOLS_BINARY_DIR}
)
if (NOT FILECHECK_EXE)
  find_program(FILECHECK_EXE FileCheck)
endif()
if (NOT FILECHECK_EXE)
  message(WARNING "FileCheck executable not found; LPN tests disabled")
  return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(LPN_OPT_EXE $<TARGET_FILE:lpn-opt>)
set(LPN_TEST_DRIVER ${CMAKE_CURRENT_SOURCE_DIR}/run_filecheck.py)

function(add_lpn_filecheck_test name input)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs TOOL_ARGS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(
    NAME ${name}
    COMMAND ${Python3_EXECUTABLE} ${LPN_TEST_DRIVER}
            --tool ${LPN_OPT_EXE}
            --filecheck ${FILECHECK_EXE}
            --input ${input}
            ${ARG_TOOL_ARGS}
  )
endfunction()

add_lpn_filecheck_test(
  lpn-normalize-delays
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/normalize-delays.mlir
  TOOL_ARGS --tool-arg=--lpn-normalize-delays
)

add_lpn_filecheck_test(
  lpn-fuse-private-places
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/fuse-private-places.mlir
  TOOL_ARGS --tool-arg=--lpn-fuse-private-places
)

add_lpn_filecheck_test(
  lpn-retain-observables
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/retain-observables.mlir
  TOOL_ARGS --tool-arg=--lpn-retain-observables
)
