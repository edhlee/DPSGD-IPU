# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.utils import ExecutionProfileType


def get_config(prng=False,
               ipu_id=-1,
               shards=1,
               number_of_replicas=1,
               max_cross_replica_buffer_size=50*1024*1024,
               merge_infeed_io_copies=True,
               fp_exceptions=True,
               half_partials=False,
               conv_dithering=False,
               conv_output=False,
               enable_recomputation=False,
               seed=None,
               profile=None,
               availableMemoryProportion=None,
               stable_norm=False,
               internalExchangeOptimisationTarget=None,
               num_io_tiles=0,
               number_of_distributed_batch_norm_replicas=1,
               min_remote_tensor_size=128,
               compile_only=False,
               nanoo=True,
               ):
    """Builds ipu_options"""

    profile_exec_modes = {"NO_PROFILE": ExecutionProfileType.NO_PROFILE,
                          "TILE_PROFILE": ExecutionProfileType.TILE_PROFILE,
                          "DEVICE_PROFILE": ExecutionProfileType.DEVICE_PROFILE,
                          "IPU_PROFILE": ExecutionProfileType.IPU_PROFILE}

    config = utils.create_ipu_config(merge_infeed_io_copies=merge_infeed_io_copies,
                                     always_rearrange_copies_on_the_host=False,
                                     profiling=profile is not None,
                                     scheduler_selection="ShortestPath",
                                     profile_execution=profile_exec_modes[profile] if profile else None)

    config = utils.set_optimization_options(config,
                                            minimum_remote_tensor_size=min_remote_tensor_size,
                                            max_cross_replica_sum_buffer_size=max_cross_replica_buffer_size)

    if ipu_id == -1:
        config = utils.auto_select_ipus(config, number_of_replicas*shards)
    else:
        config = utils.select_ipus(config, [ipu_id])
    config = utils.set_compilation_options(config, {
        "device.clearAtomicFlagAfterExchange": "false",
        "prng.enable": "true" if prng else "false",
        "target.deterministicWorkers": "false" if seed is None else "portable",
    })

    if internalExchangeOptimisationTarget is not None:
        config = utils.set_compilation_options(config, {
            "opt.internalExchangeOptimisationTarget": internalExchangeOptimisationTarget
        })

    if num_io_tiles != 0:
        config = utils.set_io_tile_options(
            config, place_ops_on_io_tiles=True, num_io_tiles=num_io_tiles)

    if availableMemoryProportion is not None:
        config = utils.set_convolution_options(config, {
            "availableMemoryProportion": str(availableMemoryProportion)
        })

    if half_partials:
        config = utils.set_convolution_options(config, {
            "partialsType": 'half'
        })
        config = utils.set_matmul_options(config, {
            "partialsType": 'half'
        })

    if conv_dithering:
        config = utils.set_convolution_options(config, {
            "enableConvDithering": "true"
        })

    if conv_output:
        config = utils.set_convolution_options(config, {
            "gatherConvOutput": "true"
        })

    if stable_norm:
        config = utils.set_norm_options(config, use_stable_statistics=True)

    if enable_recomputation:
        utils.set_recomputation_options(config, allow_recompute=True)

    config = utils.set_floating_point_behaviour_options(config, inv=fp_exceptions, div0=fp_exceptions,
                                                        oflo=fp_exceptions, esr=prng, nanoo=nanoo)

    config = utils.set_norm_options(
        config,
        experimental_distributed_batch_norm_replica_group_size=number_of_distributed_batch_norm_replicas)

    if compile_only:
        utils.set_ipu_connection_type(
            config,
            utils.DeviceConnectionType.NEVER,
            ipu_version=2,
            enable_remote_buffers=True)

    return config
