##
## Tests to verify geometric multigrid operations
##

set( library_name HDK_TestGeometricMultiGrid )

# Add a library and its source files.
add_library( ${library_name} SHARED
    HDK_GeometricMultiGridOperations.cpp
    HDK_GeometricMultiGridOperations.h
    HDK_GeometricMultiGridPoissonSolver.cpp
	HDK_GeometricMultiGridPoissonSolver.h
    HDK_TestGeometricMultiGrid.cpp
    HDK_TestGeometricMultiGrid.h
)

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries( ${library_name} Houdini )

# Include ${CMAKE_CURRENT_BINARY_DIR} for the generated header.
target_include_directories( ${library_name} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target( ${library_name} )
