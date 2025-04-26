cmake_minimum_required(VERSION 3.10)
project(Diffusion CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/Source)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Bin)

# Define all available programs
set(PROGRAMS original non-parallel non-parallel-blocking non-parallel-float non-parallel-fb parallel-omp parallel-mpi parallel-mpiomp)
set(MPI_PROGRAMS non-parallel-blocking non-parallel-fb parallel-mpi parallel-mpiomp)

# Create options for enabling each program
foreach(prog ${PROGRAMS})
    option(BUILD_${prog} "Build ${prog}" OFF)
endforeach()

# Global options
option(PROFILE "Enable profile generation (-fprofile-generate)" OFF)
option(COVERAGE "Enable test coverage (-fprofile-arcs -ftest-coverage)" OFF)
option(VALGRIND "Enable Valgrind Memcheck" OFF)

# Allow users to set the optimization level (default: -O2)
set(OPTIMIZATION_LEVEL "-O2" CACHE STRING "Optimization level (-O0, -O1, -O2, -O3, -Ofast)")

# Set appropriate flags
set(PROFILE_FLAGS "")
if(PROFILE)
    set(PROFILE_FLAGS "-fprofile-generate=reports")
elseif(COVERAGE)
    set(PROFILE_FLAGS "-fprofile-arcs -ftest-coverage --coverage")
endif()

foreach(prog ${PROGRAMS})
    if(BUILD_${prog})
        add_executable(${prog} ${SOURCE_DIR}/${prog}.cpp)

        set_target_properties(${prog} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Bin")

        # Always link OpenMP
        target_link_libraries(${prog} PRIVATE OpenMP::OpenMP_CXX)

        # Link MPI only if this program needs it
        list(FIND MPI_PROGRAMS ${prog} MPI_INDEX)
        if (NOT ${MPI_INDEX} EQUAL -1)
            target_include_directories(${prog} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
            target_link_libraries(${prog} PRIVATE ${MPI_CXX_LIBRARIES})

            if(MPI_CXX_COMPILE_FLAGS)
                set_target_properties(${prog} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
            endif()
            if(MPI_CXX_LINK_FLAGS)
                set_target_properties(${prog} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
            endif()
        endif()

        # Apply optimization level
        target_compile_options(${prog} PRIVATE ${OPTIMIZATION_LEVEL})

        # Apply profile OR coverage flags
        if(PROFILE OR COVERAGE)
            target_compile_options(${prog} PRIVATE ${PROFILE_FLAGS})
            target_link_options(${prog} PRIVATE ${PROFILE_FLAGS})
        endif()

        # Add a Valgrind test target
        if(VALGRIND)
            add_custom_target(
                valgrind_${prog}
                COMMAND valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=valgrind_${prog}.log ./Bin/${prog}
                DEPENDS ${prog}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMENT "Running Valgrind on ${prog}..."
                VERBATIM
            )
        endif()
    endif()
endforeach()
