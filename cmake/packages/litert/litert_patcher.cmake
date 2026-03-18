# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


include("${LITERTLM_MODULES_DIR}/utils.cmake")

set(ROOT_LIST "${LITERT_SRC_DIR}/CMakeLists.txt")
set(LITERTLM_LITERT_SHIM_PATH "${LITERT_PACKAGE_DIR}/litert_shims.cmake")

patch_file_content("${ROOT_LIST}" 
    "project(LiteRT VERSION 1.4.0 LANGUAGES CXX C)"
    "project(LiteRT VERSION 1.4.0 LANGUAGES CXX C)\ninclude(${LITERTLM_LITERT_SHIM_PATH})"
    FALSE
)

patch_file_content("${ROOT_LIST}" "# Add TFLite as a subdirectory" "# Add TFLite as a subdirectory\nif(FALSE)" FALSE)

patch_file_content("${ROOT_LIST}" "find_package(absl REQUIRED)" "find_package(absl REQUIRED)\nendif()" FALSE)

patch_file_content("${ROOT_LIST}" "add_subdirectory(compiler_plugin)" "add_subdirectory(compiler)" FALSE)


set(LITERTLM_BYPASS_PATHS
    "${LITERT_SRC_DIR}/third_party/tensorflow/CMakeLists.txt"
    "${LITERT_SRC_DIR}/tflite/CMakeLists.txt"
    "${LITERT_SRC_DIR}/tflite/tools/cmake/CMakeLists.txt"
)

foreach(TARGET_PATH ${LITERTLM_BYPASS_PATHS})
    get_filename_component(TARGET_DIR "${TARGET_PATH}" DIRECTORY)
    if(NOT EXISTS "${TARGET_DIR}")
        file(MAKE_DIRECTORY "${TARGET_DIR}")
    endif()

    message(STATUS "[LiteRTLM] Bypassing conflicting build path: ${TARGET_PATH}")
    file(WRITE "${TARGET_PATH}" "# Path bypassed by LiteRT-LM to prevent dependency collisions.\n")
endforeach()


file(GLOB_RECURSE ALL_CMAKELISTS "${LITERT_SRC_DIR}/*.cmake" "${LITERT_SRC_DIR}/**/CMakeLists.txt")

foreach(C_FILE ${ALL_CMAKELISTS})
    if("${C_FILE}" STREQUAL "${ROOT_LIST}")
        continue()
    endif()
    patch_file_content("${C_FILE}" "absl::[a-zA-Z0-9_]+" "LiteRTLM::absl::shim" TRUE)

    patch_file_content("${C_FILE}" "[^\" ]*/_deps/flatbuffers-build/libflatbuffers.a" "LiteRTLM::flatbuffers::flatbuffers" TRUE)

    patch_file_content("${C_FILE}" "flatbuffers-build/libflatbuffers.a" "LiteRTLM::flatbuffers::flatbuffers" FALSE)

    patch_file_content("${C_FILE}" "TFLITE_FLATBUFFERS_LIB" "LiteRTLM::flatbuffers::flatbuffers" FALSE)

    patch_file_content("${C_FILE}" "find_program\\(FLATC_EXECUTABLE[^\\)]+\\)" "# [LiteRTLM] Suppressed: Using Global Shim" TRUE)

    patch_file_content("${C_FILE}" "set\\(FLATC_EXECUTABLE \\$<TARGET_FILE:flatc>\\)" "set(FLATC_EXECUTABLE flatc)" TRUE)

    patch_file_content("${C_FILE}" "FetchContent_Declare\\([^\\)]+\\)" "# [LiteRTLM] Suppressed: External fetch prohibited" TRUE)

    patch_file_content("${C_FILE}" "FetchContent_MakeAvailable\\([^\\)]+\\)" "# [LiteRTLM] Suppressed: Using Global Manifest" TRUE)

endforeach()


patch_file_content("${LITERT_SRC_DIR}/runtime/compiled_model.cc"
    " return litert_cpu_buffer_requirements"
    "return litert::Expected<const LiteRtTensorBufferRequirementsT*>(litert_cpu_buffer_requirements)"
    FALSE
)

# Notice the \\( and \\)
patch_file_content("${LITERT_SRC_DIR}/c/litert_environment.cc"
    "env->SetGpuEnvironment\\(std::move\\(gpu_env\\)\\)\\);[ \n\r]*\\}"
    "env->SetGpuEnvironment(std::move(gpu_env)));\n  }\n#endif"
    TRUE
)

patch_file_content("${LITERT_SRC_DIR}/cc/internal/litert_runtime_builtin.cc"
    ".litert_gpu_environment_create = LiteRtGpuEnvironmentCreate,"
    "#if !defined(LITERT_DISABLE_GPU)\n  .litert_gpu_environment_create = LiteRtGpuEnvironmentCreate,\n#else\n  .litert_gpu_environment_create = nullptr,\n#endif"
    FALSE
)


set(V_LIST "${LITERT_SRC_DIR}/vendors/CMakeLists.txt")
if(EXISTS "${V_LIST}")
    file(READ "${V_LIST}" V_CONTENT)
    string(FIND "${V_CONTENT}" "if(VENDOR STREQUAL \"MediaTek\")" START_POS)

    if(NOT START_POS EQUAL -1)
        string(FIND "${V_CONTENT}" "add_custom_target(mediatek_schema_gen" ANCHOR_POS)

        if(NOT ANCHOR_POS EQUAL -1)
            message(STATUS "[LITERTLM] Decoupling Vendor Dependencies...")
            string(SUBSTRING "${V_CONTENT}" ${ANCHOR_POS} -1 POST_ANCHOR)
            string(FIND "${POST_ANCHOR}" "endif()" ENDIF_REL_POS)
            math(EXPR END_POS "${ANCHOR_POS} + ${ENDIF_REL_POS} + 7")
            string(SUBSTRING "${V_CONTENT}" 0 ${START_POS} PRE_BLOCK)
            string(SUBSTRING "${V_CONTENT}" ${END_POS} -1 POST_BLOCK)
            set(INJECTION "\n# [LITERTLM] MediaTek Logic Virtualized\ninclude(\"${VENDOR_SHIM_PATH}\")\n")
            file(WRITE "${V_LIST}" "${PRE_BLOCK}${INJECTION}${POST_BLOCK}")
        endif()
    endif()
endif()


message(STATUS "[LiteRTLM] Generating build_config.h...")
set(LITERT_GEN_DIR "${LITERT_SRC_DIR}/build_common")

if(NOT EXISTS "${LITERT_GEN_DIR}")
    file(MAKE_DIRECTORY "${LITERT_GEN_DIR}")
endif()

cmake_to_c_bool(LITERT_BUILD_CONFIG_DISABLE_GPU_VAL C_DISABLE_GPU)
cmake_to_c_bool(LITERT_BUILD_CONFIG_DISABLE_NPU_VAL C_DISABLE_NPU)

set(BUILD_CONFIG_CONTENT "/* Generated by LiteRTLM Patcher */
#ifndef LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_

#define LITERT_BUILD_CONFIG_DISABLE_GPU ${C_DISABLE_GPU}
#define LITERT_BUILD_CONFIG_DISABLE_NPU ${C_DISABLE_NPU}

#endif  /* LITE_RT_BUILD_COMMON_BUILD_CONFIG_H_ */\n")

file(WRITE "${LITERT_GEN_DIR}/build_config.h" "${BUILD_CONFIG_CONTENT}")


set(LAYOUT_HDR "${TFLITE_SRC_DIR}/../litert/cc/litert_layout.h")

if(EXISTS "${LAYOUT_HDR}")
    message(STATUS "[LITERTLM PATCHER] Neutralizing constexpr violation in litert_layout.h")

    file(READ "${LAYOUT_HDR}" CONTENT)
    string(REPLACE "constexpr LiteRTLayout" "inline LiteRTLayout" MODIFIED_CONTENT "${CONTENT}")

    if(NOT "${CONTENT}" STREQUAL "${MODIFIED_CONTENT}")
        file(WRITE "${LAYOUT_HDR}" "${MODIFIED_CONTENT}")
        message(STATUS "[LiteRTLM PATCHER] Successfully converted constexpr constructor to inline.")
    else()
        message(WARNING "[LiteRTLM PATCHER] Patch target not found in litert_layout.h. Check version compatibility.")
    endif()
endif()

message(STATUS "[LiteRTLM] Patching Phase Complete.")
