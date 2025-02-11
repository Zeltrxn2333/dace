import copy
import itertools
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Type, Any, Union


import cupy
import dace
from dace.transformation.auto_tile import auto_tile_util
from dace.transformation.auto_tile.add_thread_block_map import AddThreadBlockMap
from dace.transformation.auto_tile.thread_coarsening import ThreadCoarsening
from dace.transformation.auto_tile.explicit_memory_move import ExplicitMemoryMove
from dace.transformation.auto_tile.block_tiling import BlockTiling
from dace.transformation.auto_tile.remainder_loop import RemainderLoop
from dace.sdfg.analysis.cutout import SDFGCutout


def clean_cache():
    script_directory = os.getcwd()
    cache_dir = Path(f"{script_directory}/.dacecache")
    print(f"Clean {script_directory}/.dacecache")
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)


def copy_sub_scope(state: dace.sdfg.SDFGState, scope_entry: dace.nodes.MapEntry):
    nn = []
    for n in state.bfs_nodes(scope_entry):
        if n == state.exit_node(scope_entry):
            break
        nn.append(n)

    cut_sdfg = SDFGCutout.singlestate_cutout(state, *nn)
    return cut_sdfg


def find_node_by_cond(state, start_map_entry, cond):
    s = set([start_map_entry])
    while s:
        n = s.pop()
        if n != start_map_entry and cond(n):
            return n
        if n != state.exit_node(start_map_entry):
            s = s.union([v for _, _, v, _, _ in state.out_edges(n)])
    return None


def find_node_in_state_by_cond(state, cond):
    for n in state.nodes():
        if cond(n):
            return n
    return None


def find_nodes_by_cond(state, start_map_entry, cond):
    s = set([start_map_entry])
    ret = set()
    while s:
        n = s.pop()
        if n != start_map_entry and cond(n):
            ret.add(n)
        if n != state.exit_node(start_map_entry):
            s = s.union([v for _, _, v, _, _ in state.out_edges(n)])
    return list(ret)


def find_state_by_cond(sdfg, cond):
    for n in sdfg.states():
        if cond(n):
            return n
    return None


def get_ref_kernel_nodes_and_edges(state, kernel_entry):
    kernel_nodes = set()
    kernel_nodes_to_visit = [kernel_entry]
    kernel_edges = set()
    visited_node_guids = set()

    while kernel_nodes_to_visit:
        n = kernel_nodes_to_visit.pop(0)
        if n.guid in visited_node_guids:
            continue
        visited_node_guids.add(n.guid)
        kernel_nodes.add(n)

        kernel_edges = kernel_edges.union(state.out_edges(n))
        kernel_edges = kernel_edges.union(state.in_edges(n))

        if n != state.exit_node(kernel_entry):
            for _, _, v, _, _ in state.out_edges(n) + state.in_edges(n):
                if not v.guid in visited_node_guids:
                    kernel_nodes_to_visit.append(v)

    return (kernel_nodes, kernel_edges)


def validate_and_pad_params_to_three(params):
    validated_params = []
    for param in params:
        if len(param) < 3:
            padded_param = param + (1,) * (3 - len(param))
            validated_params.append(padded_param)
        elif len(param) == 3:
            validated_params.append(param)
        else:
            raise ValueError(
                f"Tuple {param} has length greater than 3, which is not allowed."
            )
    return validated_params


def _tile(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    entry: dace.nodes.EntryNode,
    work_on_copy: bool,
    memory_tiling_parameters: List[Tuple[int]],
    thread_coarsening_parameters: List[Tuple[int]],
    thread_block_parameters: List[Tuple[int]],
    apply_explicit_memory_transfers: List[bool],
    apply_remainder_loop: List[bool],
    inputs: Dict[Type[str], Any],
    re_apply: bool,
    verbose: bool,
    verify: bool,
    call_id: int,
):
    if not re_apply:
        raise NotImplementedError("Not re-applying is not implemeneted for tiling yet")

    # Copy kernel as a single state SDFG if we are working on the copy
    if work_on_copy:
        _kernel_sdfg = copy_sub_scope(state, entry)
        _kernel_sdfg.name = f"{sdfg.name}_auto_tiled_{call_id}"
        auto_tile_util.convert_inputs_to_gpu_storage(_kernel_sdfg)
        auto_tile_util.set_transient(_kernel_sdfg)
        auto_tile_util.convert_inputs_to_gpu_storage(_kernel_sdfg)
        auto_tile_util.set_transient(_kernel_sdfg)
        _kernel_state = _kernel_sdfg.nodes()[0]
        _kernel_entry = find_node_in_state_by_cond(
            _kernel_state,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
            and n.guid == entry.guid,
        )
        _kernel_exit = _kernel_state.exit_node(_kernel_entry)
        output_name = None
        for oe in _kernel_state.out_edges(_kernel_exit):
            if isinstance(oe.dst, dace.nodes.AccessNode):
                output_name = oe.dst.data
                break

        if output_name is None:
            raise Exception("The output name could not be deduced")

        copy_inputs = copy.deepcopy(inputs)
        non_transformed_time = auto_tile_util.run_and_measure_time(
            _kernel_sdfg, copy_inputs
        )
        output_from_non_transformed = copy_inputs[output_name]
        # Clean memory we do not need anymore
        for key in list(copy_inputs.keys()):
            if key != output_name:
                del copy_inputs[key]

        # Unset GPU events
        for node in _kernel_state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.instrument = dace.dtypes.InstrumentationType.No_Instrumentation

        kernel_work_maps = find_nodes_by_cond(
            _kernel_state,
            _kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.Sequential,
        )

        # No sequential map means no memory tiling (but memory will still be moved)
        if len(kernel_work_maps) == 0:
            memory_tiling_parameters = [(1,)]

    else:
        _kernel_sdfg = sdfg
        _kernel_state = state
        _kernel_entry = entry

    combinations = list(
        itertools.product(
            memory_tiling_parameters,
            validate_and_pad_params_to_three(thread_coarsening_parameters),
            validate_and_pad_params_to_three(thread_block_parameters),
            apply_explicit_memory_transfers,
            apply_remainder_loop,
        )
    )
    if not work_on_copy:
        if len(combinations) != 1:
            raise Exception(
                "If applying to the original sdfg (work_on_copy=False) then only one combination must be provided"
            )

    best_config = None
    best_time = None
    for i, current_config in enumerate(combinations):
        # We need to copy this sdfg if we are working in the copy as we apply transformations
        (
            memory_tiling_params,
            thread_coarsening_param,
            thread_block_param,
            apply_explicit_memory_transfer_param,
            apply_remainder_loop_param,
        ) = current_config

        if work_on_copy:
            kernel_sdfg = copy.deepcopy(_kernel_sdfg)
            kernel_sdfg.name = f"{kernel_sdfg.name}_c{i}"
            kernel_sdfg_nodes = kernel_sdfg.nodes()
            if len(kernel_sdfg_nodes) != 1:
                raise Exception("Extracted kernel should have only one state")
            kernel_state = kernel_sdfg_nodes[0]
            kernel_entry = find_node_in_state_by_cond(
                kernel_state,
                lambda n: isinstance(n, dace.nodes.MapEntry)
                and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
                and n.guid == _kernel_entry.guid,
            )
        else:
            kernel_sdfg = _kernel_sdfg
            kernel_state = _kernel_state
            kernel_entry = _kernel_entry
        # else: we do not need to do anything

        is_assign_kernel = len(kernel_state.in_edges(kernel_entry)) == 0

        AddThreadBlockMap.apply_to(
            sdfg=kernel_sdfg,
            verify=True,
            map_entry=kernel_entry,
            options={
                "thread_block_size_x": thread_block_param[0],
                "thread_block_size_y": thread_block_param[1],
                "thread_block_size_z": thread_block_param[2],
            },
        )
        # Need to restore maps after each time
        kernel_entry = kernel_state.entry_node(kernel_entry)
        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == "ThreadBlockMap",
        )
        if thread_block_map_entry is None:
            raise Exception(
                "ThreadBlock Map could not be found after applying threadblock map transformation"
            )

        ThreadCoarsening.apply_to(
            sdfg=kernel_sdfg,
            options={
                "tile_size_x": thread_coarsening_param[0],
                "tile_size_y": thread_coarsening_param[1],
                "tile_size_z": thread_coarsening_param[2],
            },
            verify=True,
            device_map_entry=kernel_entry,
            thread_block_map_entry=thread_block_map_entry,
        )
        work_maps = find_nodes_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.label != "KernelEntryMap"
            and n.label != "ThreadCoarsenedMap"
            and n.label != "ThreadBlockMap",
        )
        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == "ThreadBlockMap",
        )
        if len(work_maps) > 1:
            raise NotImplementedError(
                "Memory tiling (Tiling of Work Maps) more than once is currently not implemented"
            )

        # Need to check the "assign-kernel" variant
        # Having input should work
        if not is_assign_kernel:
            for i in range(len(work_maps)):
                work_map_entry: dace.nodes.MapEntry = work_maps[i]
                work_map_tile = memory_tiling_params[i % len(memory_tiling_params)]

                # If the passed memory tiling parameter is less than the map dimension, pad
                # If it longer, then take the first elements
                tuple_size_needed = len(work_map_entry.map.range)
                work_map_tile = work_map_tile[:tuple_size_needed] + (1,) * (
                    tuple_size_needed - len(work_map_tile)
                )

                BlockTiling.apply_to(
                    sdfg=kernel_sdfg,
                    options={"block_tile_sizes": work_map_tile},
                    verify=True,
                    thread_block_map_entry=thread_block_map_entry,
                    sequential_map_entry=work_map_entry,
                )
            thread_block_map_entry = find_node_by_cond(
                kernel_state,
                kernel_entry,
                lambda n: isinstance(n, dace.nodes.MapEntry)
                and n.map.label == "ThreadBlockMap",
            )
            if apply_explicit_memory_transfer_param[0]:
                thread_block_map_entry = find_node_by_cond(
                    kernel_state,
                    kernel_entry,
                    lambda n: isinstance(n, dace.nodes.MapEntry)
                    and n.map.label == "ThreadBlockMap",
                )
                ExplicitMemoryMove.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    device_map_entry=kernel_entry,
                    thread_block_map_entry=thread_block_map_entry,
                    map_entry=thread_block_map_entry,
                    options={
                        "memory_location": dace.dtypes.StorageType.GPU_Shared,
                        "tiles_evenly": apply_explicit_memory_transfer_param[1],
                    },
                )

        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == "ThreadBlockMap",
        )
        if apply_remainder_loop_param:
            first_inner_work_map = find_node_by_cond(
                kernel_state,
                thread_block_map_entry,
                lambda n: isinstance(n, dace.nodes.MapEntry)
                and n.map.label.startswith("InnerWorkMap"),
            )
            if len(work_maps) > 0:
                RemainderLoop.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    inner_work_map_entry=first_inner_work_map,
                )
            else:
                thread_coarsened_map = find_node_by_cond(
                    kernel_state,
                    thread_block_map_entry,
                    lambda n: isinstance(n, dace.nodes.MapEntry)
                    and n.map.label.startswith("ThreadCoarsenedMap"),
                )
                RemainderLoop.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    inner_work_map_entry=thread_coarsened_map,
                )

        time = None
        if work_on_copy and verify:
            # Check shrmem compile limit
            shr_mem_needed = 0
            for arr_name, arr in kernel_sdfg.arrays.items():
                if arr.storage == dace.dtypes.StorageType.GPU_Shared:
                    shr_mem_needed += arr.total_size * arr.dtype.bytes

            if shr_mem_needed >= 48 * 1024:
                print(
                    f"Kernel uses too much shared memory for config {current_config}, skipping."
                )
            else:
                copy_inputs_2 = copy.deepcopy(inputs)
                time = auto_tile_util.run_and_measure_time(kernel_sdfg, copy_inputs_2)
                output_from_transformed = copy_inputs_2[output_name]

                are_close = cupy.allclose(
                    output_from_transformed,
                    output_from_non_transformed,
                    rtol=1e-3,
                    atol=1e-5,
                )

                # Clean memory we do not need anymore
                for key in list(copy_inputs_2.keys()):
                    del copy_inputs_2[key]

                if not are_close:
                    raise Exception("Numerical verification failed.")

                if best_time is None or time < best_time:
                    best_config = current_config
                    best_time = time

                print(f"Transformed SDFG: {time:.10f} ms")
                print(f"Current config: {current_config}, best config: {best_config}")
                print(f"Non-transformed SDFG: {non_transformed_time:.10f} ms")

    return best_config


def _tile_search(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    entry: dace.nodes.EntryNode,
    inputs: Dict[Type[str], Any],
    re_apply: bool,
    verbose: bool,
    verify: bool,
    call_id: int,
):
    if not re_apply:
        raise NotImplementedError("Not re-applying is not implemeneted for tiling yet")

    # Copy kernel as a single state SDFG if we are working on the copy

    _kernel_sdfg = copy_sub_scope(state, entry)
    _kernel_sdfg.name = f"{sdfg.name}_auto_tiled_{call_id}"
    auto_tile_util.convert_inputs_to_gpu_storage(_kernel_sdfg)
    auto_tile_util.set_transient(_kernel_sdfg)
    auto_tile_util.convert_inputs_to_gpu_storage(_kernel_sdfg)
    auto_tile_util.set_transient(_kernel_sdfg)
    _kernel_state = _kernel_sdfg.nodes()[0]
    _kernel_entry = find_node_in_state_by_cond(
        _kernel_state,
        lambda n: isinstance(n, dace.nodes.MapEntry)
        and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
        and n.guid == entry.guid,
    )
    _kernel_exit = _kernel_state.exit_node(_kernel_entry)
    output_name = None
    for oe in _kernel_state.out_edges(_kernel_exit):
        if isinstance(oe.dst, dace.nodes.AccessNode):
            output_name = oe.dst.data
            break

    if output_name is None:
        raise Exception("The output name could not be deduced")

    copy_inputs = copy.deepcopy(inputs)
    non_transformed_time = auto_tile_util.run_and_measure_time(
        _kernel_sdfg, copy_inputs
    )
    output_from_non_transformed = copy_inputs[output_name]
    # Clean memory we do not need anymore
    for key in list(copy_inputs.keys()):
        if key != output_name:
            del copy_inputs[key]

    output_arr = _kernel_sdfg.arrays[output_name]
    dimensions = len(output_arr.shape)

    # Unset GPU events
    for node in _kernel_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            node.instrument = dace.dtypes.InstrumentationType.No_Instrumentation

    kernel_work_maps = find_nodes_by_cond(
        _kernel_state,
        _kernel_entry,
        lambda n: isinstance(n, dace.nodes.MapEntry)
        and n.map.schedule == dace.dtypes.ScheduleType.Sequential,
    )


    # We start with (1, 1, 1) for thread coarsening and (32, 1, 1) for thread block size
    # And (8, ) for work map tile size.
    assert(dimensions == 2)
    tblock_sizes = [(32, 4, 1), (32, 8, 1), (16, 16, 1), (32, 16, 1)]

    apply_remainder_loop_param = True
    apply_explicit_memory_transfer_param = (True, False)

    tried_combinations = []

    increment_thread_coarsening = False
    increment_tblock = True
    increment_mem = False

    cur_dim = 0

    def next_combination(cur_dim, tried_combinations):
        if tried_combinations == []:
            return (
                (1, 1, 1),
                (32, 1, 1),
                (8,)
            )
        elif not increment_tblock and not increment_thread_coarsening and not increment_mem:
            return None
        else:
            (
             current_thread_coarsening_size,
             current_thread_block_size,
             current_explicit_memory_move_size,
            ) = tried_combinations[-1][0]

            if increment_thread_coarsening:
                l = list(current_thread_coarsening_size)
                l[cur_dim] += 2 if l[cur_dim] != 1 else 1
                current_thread_coarsening_size = tuple(l)
            elif increment_tblock:
                l = tblock_sizes.pop(0)
                current_thread_block_size = tuple(l)
            elif increment_mem:
                l = list(current_explicit_memory_move_size)
                if cur_dim == 0:
                    l[cur_dim] += 8
                else:
                    l[cur_dim] += 2 if l[cur_dim] != 1 else 1
                current_explicit_memory_move_size = tuple(l)

            cur_dim = cur_dim + 1
            if cur_dim == dimensions:
                cur_dim = 0

            return (
                current_thread_coarsening_size,
                current_thread_block_size,
                current_explicit_memory_move_size
            )

    i = 0
    cur_combination = next_combination(cur_dim, tried_combinations)
    while cur_combination is not None:
        print(f"Testing config: {cur_combination}")

        # We need to copy this sdfg if we are working in the copy as we apply transformations
        current_config = cur_combination
        (
            thread_coarsening_param,
            thread_block_param,
            memory_tiling_params

        ) = current_config
        apply_explicit_memory_transfer_param = (True, False)
        apply_remainder_loop_param = True

        kernel_sdfg = copy.deepcopy(_kernel_sdfg)
        kernel_sdfg.name = f"{kernel_sdfg.name}_c{i}"
        kernel_sdfg_nodes = kernel_sdfg.nodes()
        if len(kernel_sdfg_nodes) != 1:
            raise Exception("Extracted kernel should have only one state")
        kernel_state = kernel_sdfg_nodes[0]
        kernel_entry = find_node_in_state_by_cond(
            kernel_state,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.GPU_Device
            and n.guid == _kernel_entry.guid,
        )

        is_assign_kernel = len(kernel_state.in_edges(kernel_entry)) == 0

        AddThreadBlockMap.apply_to(
            sdfg=kernel_sdfg,
            verify=True,
            map_entry=kernel_entry,
            options={
                "thread_block_size_x": thread_block_param[0],
                "thread_block_size_y": thread_block_param[1],
                "thread_block_size_z": thread_block_param[2],
            },
        )
        # Need to restore maps after each time
        kernel_entry = kernel_state.entry_node(kernel_entry)
        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == "ThreadBlockMap",
        )
        if thread_block_map_entry is None:
            raise Exception(
                "ThreadBlock Map could not be found after applying threadblock map transformation"
            )

        ThreadCoarsening.apply_to(
            sdfg=kernel_sdfg,
            options={
                "tile_size_x": thread_coarsening_param[0],
                "tile_size_y": thread_coarsening_param[1],
                "tile_size_z": thread_coarsening_param[2],
            },
            verify=True,
            device_map_entry=kernel_entry,
            thread_block_map_entry=thread_block_map_entry,
        )
        work_maps = find_nodes_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.label != "KernelEntryMap"
            and n.label != "ThreadCoarsenedMap"
            and n.label != "ThreadBlockMap",
        )
        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == "ThreadBlockMap",
        )
        if len(work_maps) > 1:
            raise NotImplementedError(
                "Memory tiling (Tiling of Work Maps) more than once is currently not implemented"
            )

        # Need to check the "assign-kernel" variant
        # Having input should work
        if not is_assign_kernel:
            for i in range(len(work_maps)):
                work_map_entry: dace.nodes.MapEntry = work_maps[i]
                work_map_tile = memory_tiling_params[i % len(memory_tiling_params)]

                # If the passed memory tiling parameter is less than the map dimension, pad
                # If it longer, then take the first elements
                tuple_size_needed = len(work_map_entry.map.range)
                work_map_tile = work_map_tile[:tuple_size_needed] + (1,) * (
                    tuple_size_needed - len(work_map_tile)
                )

                BlockTiling.apply_to(
                    sdfg=kernel_sdfg,
                    options={"block_tile_sizes": work_map_tile},
                    verify=True,
                    thread_block_map_entry=thread_block_map_entry,
                    sequential_map_entry=work_map_entry,
                )
            thread_block_map_entry = find_node_by_cond(
                kernel_state,
                kernel_entry,
                lambda n: isinstance(n, dace.nodes.MapEntry)
                and n.map.label == "ThreadBlockMap",
            )
            if apply_explicit_memory_transfer_param[0]:
                thread_block_map_entry = find_node_by_cond(
                    kernel_state,
                    kernel_entry,
                    lambda n: isinstance(n, dace.nodes.MapEntry)
                    and n.map.label == "ThreadBlockMap",
                )
                ExplicitMemoryMove.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    device_map_entry=kernel_entry,
                    thread_block_map_entry=thread_block_map_entry,
                    map_entry=thread_block_map_entry,
                    options={
                        "memory_location": dace.dtypes.StorageType.GPU_Shared,
                        "tiles_evenly": apply_explicit_memory_transfer_param[1],
                    },
                )

        thread_block_map_entry = find_node_by_cond(
            kernel_state,
            kernel_entry,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.label == "ThreadBlockMap",
        )
        if apply_remainder_loop_param:
            first_inner_work_map = find_node_by_cond(
                kernel_state,
                thread_block_map_entry,
                lambda n: isinstance(n, dace.nodes.MapEntry)
                and n.map.label.startswith("InnerWorkMap"),
            )
            if len(work_maps) > 0:
                RemainderLoop.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    inner_work_map_entry=first_inner_work_map,
                )
            else:
                thread_coarsened_map = find_node_by_cond(
                    kernel_state,
                    thread_block_map_entry,
                    lambda n: isinstance(n, dace.nodes.MapEntry)
                    and n.map.label.startswith("ThreadCoarsenedMap"),
                )
                RemainderLoop.apply_to(
                    sdfg=kernel_sdfg,
                    verify=True,
                    inner_work_map_entry=thread_coarsened_map,
                )

        time = None
        if verify:
            # Check shrmem compile limit
            shr_mem_needed = 0
            for arr_name, arr in kernel_sdfg.arrays.items():
                if arr.storage == dace.dtypes.StorageType.GPU_Shared:
                    shr_mem_needed += arr.total_size * arr.dtype.bytes

            if shr_mem_needed >= 48 * 1024:
                print(
                    f"Kernel uses too much shared memory for config {current_config}, skipping."
                )
            else:
                copy_inputs_2 = copy.deepcopy(inputs)
                time = auto_tile_util.run_and_measure_time(kernel_sdfg, copy_inputs_2)
                output_from_transformed = copy_inputs_2[output_name]

                are_close = cupy.allclose(
                    output_from_transformed,
                    output_from_non_transformed,
                    rtol=1e-3,
                    atol=1e-5,
                )

                if len(tried_combinations) > 0:
                    last_config, last_time = tried_combinations[-1]
                    tried_combinations.append((current_config, time))

                    # Appending last config means backtracking we try the next config
                    if increment_tblock:
                        if len(tblock_sizes) > 0:
                            cur_dim = 0
                        else:
                            best_config =  min(tried_combinations, key=lambda x: x[1])
                            tried_combinations.append(best_config)
                            increment_tblock = False
                            cur_dim = 0
                            increment_thread_coarsening = True
                    elif last_time < time:
                        if increment_thread_coarsening:
                            cur_dim = 0
                            increment_thread_coarsening = False
                            tried_combinations.append((last_config, last_time))
                            if len(work_maps) != 0:
                                increment_mem = True
                        elif increment_mem:
                            tried_combinations.append((last_config, last_time))
                            increment_mem = False
                else:
                    tried_combinations.append((current_config, time))

                cur_combination = next_combination(cur_dim, tried_combinations)

                # Clean memory we do not need anymore
                for key in list(copy_inputs_2.keys()):
                    del copy_inputs_2[key]

                if not are_close:
                    raise Exception("Numerical verification failed.")

                print(f"Transformed SDFG: {time} ms")
                print(f"Non-transformed SDFG: {non_transformed_time} ms")

        i += 1

    best_config = min(tried_combinations, key=lambda x: x[1])
    best_config = (best_config[0][0], best_config[0][1], best_config[0][2], (True, False), True, best_config[1])
    print("Best config:", best_config)
    return best_config

# Possible future parameters:
# sdfg_peak_flops_and_mem_access: Union[Tuple[int, int], None] = None,
# machine_peak_flops_and_bandwidth: Union[Tuple[int, int], None] = None,


def auto_tile_gpu(
    sdfg: dace.SDFG,
    exhaustive_search: bool,
    memory_tiling_parameters: List[Tuple[int]],
    thread_coarsening_parameters: List[Tuple[int]],
    thread_block_parameters: List[Tuple[int]],
    apply_explicit_memory_transfers: List[bool],
    apply_remainder_loop: List[bool],
    inputs: Dict[Type[str], Any],
    device_schedule: dace.dtypes.ScheduleType = dace.dtypes.ScheduleType.GPU_Device,
    re_apply: bool = False,
    verbose: bool = False,
):
    sdfg_name = sdfg.name
    sym_dict = sdfg.symbols

    # Create report folder and file
    folder = Path(f"{sdfg_name}_report")
    filename = Path.joinpath(folder, Path(f"{sdfg_name}.report"))
    folder.mkdir(parents=True, exist_ok=True)
    tiled_sdfg_path = Path.joinpath(folder, Path(f"{sdfg_name}_auto_tiled.sdfgz"))

    # If this SDFG was tiled before, just return
    if filename.exists() and tiled_sdfg_path.exists() and not re_apply:
        return dace.SDFG.from_file(str(tiled_sdfg_path)), None

    # filename.open('w').close() if filename.exists() else filename.touch()

    # Collect Device kernels
    kernel_guids: List[Tuple[dace.sdfg.SDFGState, str]] = []
    for state in sdfg.states():
        for node in state.nodes():
            if (
                isinstance(node, dace.nodes.MapEntry)
                and node.map.schedule == device_schedule
                and (state, node.guid) not in kernel_guids
            ):
                kernel_guids.append((state, node))

    # Apply tiling one-by-one to kernels
    found_tilings = dict()
    for ii, (state, kernel_entry) in enumerate(kernel_guids):
        if exhaustive_search:
            best_config = _tile(
                sdfg=sdfg,
                state=state,
                entry=kernel_entry,
                work_on_copy=True,
                memory_tiling_parameters=memory_tiling_parameters,
                thread_coarsening_parameters=thread_coarsening_parameters,
                thread_block_parameters=thread_block_parameters,
                apply_explicit_memory_transfers=apply_explicit_memory_transfers,
                apply_remainder_loop=apply_remainder_loop,
                inputs=inputs,
                re_apply=re_apply,
                verbose=verbose,
                verify=True,
                call_id=ii,
            )
            found_tilings[(state.guid, kernel_entry.guid)] = best_config
        else:
            best_config = _tile_search(
                sdfg=sdfg,
                state=state,
                entry=kernel_entry,
                inputs=inputs,
                re_apply=re_apply,
                verbose=verbose,
                verify=True,
                call_id=ii,
            )
            found_tilings[(state.guid, kernel_entry.guid)] = tuple(list(best_config) + [(True, False), True])

        if verbose:
            print(f"Best Tiling Configuration for {kernel_entry.label}: {best_config}")

    for (state_guid, kernel_entry_guid), best_config in found_tilings.items():
        state = find_state_by_cond(sdfg, lambda n: n.guid == state_guid)
        if state is None:
            raise Exception("After auto-tiling, the state is none")
        kernel_entry = find_node_in_state_by_cond(
            state, lambda n: n.guid == kernel_entry_guid
        )
        if kernel_entry is None:
            raise Exception("After auto-tiling the kernel entry is none")
        # Create a single element list for applying the transformations

        memory_tiling = [best_config[0]]
        thread_coarsening = [best_config[1]]
        thread_block_coarsening = [best_config[2]]
        if exhaustive_search:
            explicit_memory_transfer = [best_config[3]]
            remainder_loop = [best_config[4]]
            _tile(
                sdfg=sdfg,
                state=state,
                entry=kernel_entry,
                work_on_copy=False,
                memory_tiling_parameters=memory_tiling,
                thread_coarsening_parameters=thread_coarsening,
                thread_block_parameters=thread_block_coarsening,
                apply_explicit_memory_transfers=explicit_memory_transfer,
                apply_remainder_loop=remainder_loop,
                inputs=inputs,
                re_apply=True,
                verbose=verbose,
                verify=False,
                call_id=len(kernel_guids),
            )
        else:
            _tile(
                sdfg=sdfg,
                state=state,
                entry=kernel_entry,
                work_on_copy=False,
                memory_tiling_parameters=memory_tiling,
                thread_coarsening_parameters=thread_coarsening,
                thread_block_parameters=thread_block_coarsening,
                apply_explicit_memory_transfers=[(True, False)],
                apply_remainder_loop=[True],
                inputs=inputs,
                re_apply=True,
                verbose=verbose,
                verify=False,
                call_id=len(kernel_guids),
            )

    # Add missing symbols
    for input_sym, sym in inputs.items():
        if input_sym not in sdfg.symbols and input_sym not in sdfg.arrays:
            if isinstance(sym, dace.symbolic.symbol):
                sdfg.add_symbol(input_sym, sym.dtype)
            else:
                sdfg.add_symbol(input_sym, dace.dtypes.typeclass(type(sym)))

    return sdfg, found_tilings
