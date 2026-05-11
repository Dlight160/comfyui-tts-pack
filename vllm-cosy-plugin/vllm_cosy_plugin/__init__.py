# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import ModelRegistry


def register():
    if "CosyVoice2ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "CosyVoice2ForCausalLM",
            "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
        )
