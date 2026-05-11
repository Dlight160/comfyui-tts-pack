# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="vllm-cosy-plugin",
    version="0.1",
    packages=["vllm_cosy_plugin"],
    entry_points={
        "vllm.general_plugins": [
            "register_cosyvoice = vllm_cosy_plugin:register"
        ]
    },
)
