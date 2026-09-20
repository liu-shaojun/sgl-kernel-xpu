set(GROUP_GEMM_Q6K_XE20_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/GroupGemmQ6KXe20LauncherInstance.cpp.in")
set(GROUP_GEMM_Q6K_XE20_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/group_gemm_q6k_xe20")
set(GROUP_GEMM_Q6K_XE20_INST_SRCS)
file(MAKE_DIRECTORY ${GROUP_GEMM_Q6K_XE20_GEN_DIR})

function(add_group_gemm_q6k_xe20_inst POLICY ELEMENT_A ACT_TAG)
    set(GEN_SRC "${GROUP_GEMM_Q6K_XE20_GEN_DIR}/GroupGemmQ6KXe20_inst_${POLICY}_${ACT_TAG}.cpp")
    configure_file(${GROUP_GEMM_Q6K_XE20_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND GROUP_GEMM_Q6K_XE20_INST_SRCS ${GEN_SRC})
    set(GROUP_GEMM_Q6K_XE20_INST_SRCS ${GROUP_GEMM_Q6K_XE20_INST_SRCS} PARENT_SCOPE)
endfunction()

foreach(policy
        q6_k_launch_policy_m_8_n_64
        q6_k_launch_policy_m_16_n_64
        q6_k_launch_policy_m_32_n_64
        q6_k_launch_policy_m_64_n_128
        q6_k_launch_policy_m_64_n_128_skip
        q6_k_launch_policy_m_64_n_256
        q6_k_launch_policy_m_64_n_256_skip)
    add_group_gemm_q6k_xe20_inst(${policy} "cutlass::bfloat16_t" "bf16")
    add_group_gemm_q6k_xe20_inst(${policy} "cutlass::half_t" "fp16")
endforeach()

list(APPEND device_cpp_xe20 ${GROUP_GEMM_Q6K_XE20_INST_SRCS})
