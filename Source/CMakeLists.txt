##
## Tests to verify geometric multigrid operations
##

set( library_name HDK_TestGeometricMultigrid )

# Add a library and its source files.
add_library( ${library_name} SHARED
    HDK_GeometricMultigridOperators.cpp
    HDK_GeometricMultigridOperators.h
    HDK_GeometricMultigridPoissonSolver.cpp
	HDK_GeometricMultigridPoissonSolver.h
    HDK_TestGeometricMultigrid.cpp
    HDK_TestGeometricMultigrid.h
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

##
## Pressure solver using assembly-free CG and multigrid as a preconditioner
##

set( library_name HDK_GeometricFreeSurfacePressureSolver )

# Add a library and its source files.
add_library( ${library_name} SHARED
	HDK_GeometricFreeSurfacePressureSolver.cpp
	HDK_GeometricFreeSurfacePressureSolver.h
    HDK_GeometricMultigridOperators.cpp
    HDK_GeometricMultigridOperators.h
    HDK_GeometricMultigridPoissonSolver.cpp
	HDK_GeometricMultigridPoissonSolver.h
    HDK_Utilities.cpp
    HDK_Utilities.h
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

##
## Pressure solver using assembled PCG
##

set( library_name HDK_FreeSurfacePressureSolver )

# Add a library and its source files.
add_library( ${library_name} SHARED
    HDK_FreeSurfacePressureSolver.cpp
    HDK_FreeSurfacePressureSolver.h
    HDK_Utilities.cpp
    HDK_Utilities.h
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