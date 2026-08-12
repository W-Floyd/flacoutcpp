cmake_minimum_required(VERSION 3.5...4.10)
project(flacoutcpp VERSION 1.3.0)

if(MSVC)
    # force static msvc runtime linkage (/mt for release, /mtd for debug)
    if(POLICY CMP0091)
        cmake_policy(SET CMP0091 NEW)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Without this a plain `cmake -S . -B build` compiles with no -O flag at all,
# which makes the optimizer's hot loops roughly an order of magnitude slower.
# Single-config generators only; multi-config ones pick the type at build time.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "CMAKE_BUILD_TYPE not set — defaulting to Release")
endif()

# Opt-in host-specific codegen. Worth ~7% on the exhaustive path, but produces
# binaries that may not run on other machines, so it is off by default.
option(FLACOUT_NATIVE "Optimize for the build host's CPU (not portable)" OFF)

# Opt-in Vulkan compute backend (`-G`). Off by default so the build keeps its
# single dependency; when off, src/gpu.cpp compiles to a stub and `-G` reports
# why rather than failing to link. The kernel is bit-exact with the CPU cost
# model, so a GPU build still passes bench/check.sh.
option(FLACOUT_VULKAN "Build the Vulkan compute backend (-G)" OFF)

# Skip vendoring our own copy if a FLAC target already exists — e.g. when
# flacoutcpp is add_subdirectory'd into a parent project that already
# provides one (its own libflac submodule), so the two don't collide.
if(NOT TARGET FLAC)
    set(WITH_OGG OFF CACHE BOOL "Disable Ogg support in vendored FLAC" FORCE)
    set(INSTALL_MANPAGES OFF CACHE BOOL "Don't build FLAC manpages" FORCE)
    add_subdirectory(third_party/libflac EXCLUDE_FROM_ALL)
endif()

include_directories(
    include
    third_party/libflac/include
)

# Core library (optimizer + processor + frame writer + bitwriter + library shim)
add_library(flacout_lib
    src/optimizer.cpp
    src/processor.cpp
    src/bitwriter.cpp
    src/frame_writer.cpp
    src/flacoutcpp.cpp
    src/gpu.cpp
    src/gpu_encoder.cpp
)
target_link_libraries(flacout_lib FLAC)

if(FLACOUT_VULKAN)
    find_package(Vulkan REQUIRED)
    # glslangValidator ships with the Vulkan SDK; glslc with shaderc. Either
    # will do, and Homebrew installs them separately, so accept both.
    find_program(FLACOUT_GLSLC NAMES glslc)
    find_program(FLACOUT_GLSLANG NAMES glslangValidator)
    set(SWEEP_SPV "${CMAKE_CURRENT_BINARY_DIR}/sweep.spv")
    set(SWEEP_HDR "${CMAKE_CURRENT_BINARY_DIR}/generated/sweep_spv.h")
    if(FLACOUT_GLSLC)
        set(SWEEP_CMD ${FLACOUT_GLSLC} -O --target-env=vulkan1.1
            -fshader-stage=compute "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sweep.comp"
            -o "${SWEEP_SPV}")
    elseif(FLACOUT_GLSLANG)
        set(SWEEP_CMD ${FLACOUT_GLSLANG} -V --target-env vulkan1.1 -S comp
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sweep.comp" -o "${SWEEP_SPV}")
    else()
        message(FATAL_ERROR "FLACOUT_VULKAN=ON needs glslc or glslangValidator on PATH")
    endif()

    # Embed the SPIR-V so the binary is self-contained -- no shader file to
    # ship, lose, or get out of sync with the cost model it mirrors.
    add_custom_command(
        OUTPUT "${SWEEP_HDR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND ${SWEEP_CMD}
        COMMAND ${CMAKE_COMMAND}
                -DSPV=${SWEEP_SPV} -DHDR=${SWEEP_HDR}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sweep.comp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        COMMENT "Compiling and embedding shaders/sweep.comp"
        VERBATIM)

    # Fallback variants of the same source for devices missing subgroup
    # ARITHMETIC, shaderInt64, or both (old Intel iGPUs under Mesa hasvk miss
    # both). One module per combination, so a device pays only for the feature it
    # actually lacks; src/gpu.cpp picks at device-init time. The default variant
    # above is untouched, so a capable GPU runs exactly the SPIR-V it ran before
    # these existed -- verified by comparing sweep.spv across the change.
    # name:defines:symbol, defines comma-separated -- a ';' here would be a list
    # separator and split the entry in two, which is exactly the bug this comment
    # is standing in for.
    set(SWEEP_VARIANTS
        "noarith:SWEEP_NO_ARITH:kSweepNoArithSpv"
        "noint64:SWEEP_NO_INT64:kSweepNoInt64Spv"
        "compat:SWEEP_NO_ARITH,SWEEP_NO_INT64:kSweepCompatSpv"
        "slm:SWEEP_SLM_STATE:kSweepSlmSpv"
        "compat_slm:SWEEP_NO_ARITH,SWEEP_NO_INT64,SWEEP_SLM_STATE:kSweepCompatSlmSpv")
    set(SWEEP_VAR_HDRS)
    foreach(_v ${SWEEP_VARIANTS})
        string(REPLACE ":" ";" _vparts "${_v}")
        list(GET _vparts 0 _vname)
        list(GET _vparts 1 _vdefstr)
        list(GET _vparts 2 _vsym)
        string(REPLACE "," ";" _vdefs "${_vdefstr}")
        set(_vspv "${CMAKE_CURRENT_BINARY_DIR}/sweep_${_vname}.spv")
        set(_vhdr "${CMAKE_CURRENT_BINARY_DIR}/generated/sweep_${_vname}_spv.h")
        set(_vflags)
        foreach(_d ${_vdefs})
            list(APPEND _vflags "-D${_d}")
        endforeach()
        if(FLACOUT_GLSLC)
            set(_vcmd ${FLACOUT_GLSLC} -O --target-env=vulkan1.1 ${_vflags}
                -fshader-stage=compute "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sweep.comp"
                -o "${_vspv}")
        else()
            set(_vcmd ${FLACOUT_GLSLANG} -V --target-env vulkan1.1 -S comp ${_vflags}
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sweep.comp" -o "${_vspv}")
        endif()
        add_custom_command(
            OUTPUT "${_vhdr}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
            COMMAND ${_vcmd}
            COMMAND ${CMAKE_COMMAND}
                    -DSPV=${_vspv} -DHDR=${_vhdr}
                    -DSYM=${_vsym} -DSRC=shaders/sweep.comp
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sweep.comp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
            COMMENT "Compiling and embedding shaders/sweep.comp (${_vdefs})"
            VERBATIM)
        list(APPEND SWEEP_VAR_HDRS "${_vhdr}")
    endforeach()
    # Pure-GPU pipeline (-P): one kernel per stage, each embedded the same way.
    # Names map mechanically: shaders/pg_foo.comp -> pg_foo_spv.h -> kPgFooSpv.
    set(PG_SHADERS prepare autoc levinson quant fixed sweep
                   select rice layout frame pack crc probe)
    set(PG_HDRS "")
    foreach(pg ${PG_SHADERS})
        set(_src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pg_${pg}.comp")
        set(_spv "${CMAKE_CURRENT_BINARY_DIR}/pg_${pg}.spv")
        set(_hdr "${CMAKE_CURRENT_BINARY_DIR}/generated/pg_${pg}_spv.h")
        # kPgPrepareSpv from "prepare": capitalise the first letter only.
        string(SUBSTRING "${pg}" 0 1 _first)
        string(SUBSTRING "${pg}" 1 -1 _rest)
        string(TOUPPER "${_first}" _first)
        set(_sym "kPg${_first}${_rest}Spv")
        # FLACOUT_PG_DEFS lets an ablation pass -D flags to every pg kernel
        # without editing the shader (e.g. -DNACC=4 for the sweep's accumulators).
        if(FLACOUT_GLSLC)
            set(_cmd ${FLACOUT_GLSLC} -O --target-env=vulkan1.1 ${FLACOUT_PG_DEFS}
                     -fshader-stage=compute "${_src}" -o "${_spv}")
        else()
            set(_cmd ${FLACOUT_GLSLANG} -V --target-env vulkan1.1 ${FLACOUT_PG_DEFS}
                     -S comp "${_src}" -o "${_spv}")
        endif()
        add_custom_command(
            OUTPUT "${_hdr}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
            COMMAND ${_cmd}
            COMMAND ${CMAKE_COMMAND} -DSPV=${_spv} -DHDR=${_hdr}
                    -DSYM=${_sym} -DSRC=shaders/pg_${pg}.comp
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
            DEPENDS "${_src}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
            COMMENT "Compiling and embedding shaders/pg_${pg}.comp"
            VERBATIM)
        list(APPEND PG_HDRS "${_hdr}")
    endforeach()

    # A second build of pg_autoc with the `precise` qualifiers compiled out. It
    # is the same kernel and, on a driver that does not reassociate fp32, the
    # same arithmetic -- 5x faster there for byte-identical output. Which one
    # runs is decided at init by shaders/pg_probe.comp, never by vendor name.
    set(_asrc "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pg_autoc.comp")
    set(_aspv "${CMAKE_CURRENT_BINARY_DIR}/pg_autoc_fast.spv")
    set(_ahdr "${CMAKE_CURRENT_BINARY_DIR}/generated/pg_autoc_fast_spv.h")
    if(FLACOUT_GLSLC)
        set(_acmd ${FLACOUT_GLSLC} -O --target-env=vulkan1.1 -DPG_NO_PRECISE
                  -fshader-stage=compute "${_asrc}" -o "${_aspv}")
    else()
        set(_acmd ${FLACOUT_GLSLANG} -V --target-env vulkan1.1 -DPG_NO_PRECISE
                  -S comp "${_asrc}" -o "${_aspv}")
    endif()
    add_custom_command(
        OUTPUT "${_ahdr}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND ${_acmd}
        COMMAND ${CMAKE_COMMAND} -DSPV=${_aspv} -DHDR=${_ahdr}
                -DSYM=kPgAutocFastSpv -DSRC=shaders/pg_autoc.comp
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        DEPENDS "${_asrc}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        COMMENT "Compiling and embedding shaders/pg_autoc.comp (PG_NO_PRECISE)"
        VERBATIM)
    list(APPEND PG_HDRS "${_ahdr}")

    add_custom_target(flacout_shaders
        DEPENDS "${SWEEP_HDR}" ${SWEEP_VAR_HDRS} ${PG_HDRS})
    add_dependencies(flacout_lib flacout_shaders)

    target_include_directories(flacout_lib PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(flacout_lib PUBLIC FLACOUT_HAVE_VULKAN)
    target_link_libraries(flacout_lib Vulkan::Vulkan)
endif()

# Pin floating-point contraction off.
#
# The optimizer picks LPC coefficients from a double-precision autocorrelation
# and then quantizes them to 8-15 bit integers. A last-bit difference in that
# sum almost always quantizes to the same integer, but occasionally lands one
# step over — different residuals, different cost, a different candidate wins.
# So whether the compiler fuses `a += x * y` into a single-rounding FMA is
# visible in the encoded output.
#
# Left to the compiler, that makes the bitstream a function of the toolchain:
# clang at -O3 and the same clang with -ffp-contract=off disagree on 7 of the 14
# bench/check.sh cases. For a tool whose whole purpose is deterministic optimal
# encoding, that is a defect. Pinning it also lets the autocorrelation be
# restructured for speed without perturbing results, which is worth 1.8x on
# ranked search.
#
# Costs nothing measurable — the FP work is a rounding error next to the integer
# residual and Rice loops.
if(MSVC)
    # MSVC does not contract under /fp:precise, which is also its default. Set
    # it explicitly so a project-wide /fp:fast cannot silently re-enable it.
    target_compile_options(flacout_lib PRIVATE /fp:precise)
else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-ffp-contract=off" FLACOUT_HAS_FP_CONTRACT_OFF)
    if(FLACOUT_HAS_FP_CONTRACT_OFF)
        target_compile_options(flacout_lib PRIVATE -ffp-contract=off)
    else()
        message(WARNING
            "Compiler does not accept -ffp-contract=off; encoded output may "
            "differ from other builds. See the comment in CMakeLists.txt.")
    endif()
endif()

if(FLACOUT_NATIVE AND NOT MSVC)
    include(CheckCXXCompilerFlag)
    # Apple Silicon clang wants -mcpu=native; x86 wants -march=native.
    check_cxx_compiler_flag("-mcpu=native" FLACOUT_HAS_MCPU_NATIVE)
    if(FLACOUT_HAS_MCPU_NATIVE)
        target_compile_options(flacout_lib PRIVATE -mcpu=native)
    else()
        check_cxx_compiler_flag("-march=native" FLACOUT_HAS_MARCH_NATIVE)
        if(FLACOUT_HAS_MARCH_NATIVE)
            target_compile_options(flacout_lib PRIVATE -march=native)
        endif()
    endif()
endif()

# CLI executable
add_executable(flacoutcpp src/main.cpp)
target_link_libraries(flacoutcpp flacout_lib)
target_compile_definitions(flacoutcpp PRIVATE FLACOUTCPP_VERSION="${PROJECT_VERSION}")

install(TARGETS flacoutcpp DESTINATION bin)
