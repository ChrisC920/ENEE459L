from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def _layer_parameters(ly: Layer) -> int | None:
    if ly.kind in {"relu", "add", "pool", "flatten"}:
        return 0
    if ly.kind == "bn":
        if not ly.out_shape:
            return None
        return ly.out_shape[0] * BN_PARAMS_PER_CHANNEL
    if ly.kind == "linear":
        count = ly.out_elements * ly.in_elements
        if ly.bias:
            count += ly.out_elements
        return count
    if ly.kind == "conv":
        if ly.kernel is None or not ly.in_shape or not ly.out_shape:
            return None
        channels_in = ly.in_shape[0]
        channels_out = ly.out_shape[0]
        if ly.groups < 1 or channels_in % ly.groups or channels_out % ly.groups:
            return None
        kernel = 1
        for dim in ly.kernel:
            kernel *= dim
        count = channels_out * (channels_in // ly.groups) * kernel
        if ly.bias:
            count += channels_out
        return count
    return None


def _whole(number: float | int) -> int | float:
    if isinstance(number, float) and number.is_integer():
        return int(number)
    return number


def count_parameters(graph: Graph) -> dict[str, Any]:
    source = f"{graph.name}: {len(graph)} layers, shapes from the description"
    per_layer: dict[str, int] = {}
    total = 0
    for ly in graph.layers:
        count = _layer_parameters(ly)
        if count is None:
            return unknown(source, f"{ly.name}: cannot count parameters for kind {ly.kind!r}")
        per_layer[ly.name] = count
        total += count
    return computed(
        total,
        source,
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )


# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Bytes of stored tensors: parameters plus buffers, at their own dtypes.

    Lecture 04 slide 8 gives the formula as `#Parameters × bit width` and slide
    9 spends a page on why the file on disk is not that number. Three reasons,
    two of which this function has to get right:

      * a model is not stored in one dtype. `Layer.weight_dtype` is per layer
        and a network with FP16 weights and FP32 normalisation is completely
        ordinary. Multiplying a single total by a single bit width is the
        mistake, and on these four descriptions it is worth several per cent
      * buffers are in the file. Batch norm's running statistics are two
        vectors per channel that no optimiser ever touched, and they are still
        bytes you have to ship
      * the container is in the file too — the pickle framing, the state-dict
        keys, the archive directory. This function does *not* try to model
        that, and it says so in `container_overhead_excluded` rather than
        quietly letting the caller assume it did

    Returns a `computed` finding whose value is bytes, with the per-dtype
    breakdown that makes the first bullet checkable.
    """
    source = f"{graph.name}: per-layer dtypes, buffers at {BUFFER_DTYPE}"
    params = count_parameters(graph)
    if not is_answered(params):
        return unknown(source, params.get("detail", "parameter count is unknown"))

    per_layer: dict[str, int | float] = {}
    per_dtype: dict[str, float] = {}
    buffer_bytes = 0.0
    total = 0.0
    for ly in graph.layers:
        count = params["per_layer"][ly.name]
        try:
            param_bytes = count * dtype_bytes(ly.weight_dtype) if count else 0
        except KeyError:
            return unknown(source, f"{ly.name}: unknown weight dtype {ly.weight_dtype!r}")
        layer_buffers = 0.0
        if ly.kind == "bn":
            try:
                layer_buffers = (
                    ly.out_shape[0] * BN_BUFFERS_PER_CHANNEL * dtype_bytes(BUFFER_DTYPE)
                )
            except KeyError:
                return unknown(source, f"unknown buffer dtype {BUFFER_DTYPE!r}")
        layer_bytes = param_bytes + layer_buffers
        per_layer[ly.name] = _whole(layer_bytes)
        total += layer_bytes
        buffer_bytes += layer_buffers
        if param_bytes:
            per_dtype[ly.weight_dtype] = per_dtype.get(ly.weight_dtype, 0.0) + float(param_bytes)
        if layer_buffers:
            per_dtype[BUFFER_DTYPE] = per_dtype.get(BUFFER_DTYPE, 0.0) + float(layer_buffers)

    return computed(
        _whole(total),
        source,
        per_layer=per_layer,
        per_dtype=per_dtype,
        buffer_bytes=float(buffer_bytes),
        container_overhead_excluded=True,
        note="not the size of the file on disk; see the handout, Stage A step 3",
    )

# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def count_activations(graph: Graph) -> dict[str, Any]:
    """Total and peak activation footprint, in elements and in bytes.

    UNC COMP 790-150 Lec 2 p. 70 gives AlexNet as total 932,264 and peak
    440,928, and the two numbers answer two different questions. Total is what
    the whole forward pass produced. Peak is how much had to be resident at
    once, and peak is the one that decides whether the model runs.

    Peak is not `max(out_elements)`. Three things make it larger than that:

      * a layer's input is still resident while its output is being written.
        The live set at layer *i* contains both
      * a tensor consumed by a later layer stays resident in between. `add`
        layers name two inputs in `Layer.reads`, and the earlier one has been
        sitting in memory across every layer of the block. This is the residual
        connection and it is the single largest contributor to peak in
        ResNet-shaped networks
      * the network's own input is a tensor too

    The implementation is a liveness pass: work out the last layer that reads
    each tensor, then walk forward keeping a live set and taking the maximum of
    its total size. Anything simpler than that is wrong on any graph with a
    skip connection, and it is wrong quietly, in the direction that says the
    model fits.

    Returns a `computed` finding whose value is peak *bytes*, because bytes are
    what a memory budget is denominated in, with elements and the layer where
    the peak occurs alongside.
    """
    source = f"{graph.name}: liveness over {len(graph)} layers, input included"
    input_name = "__input__"
    elements: dict[str, int] = {}
    nbytes: dict[str, float] = {}

    input_elements = 1
    for dim in graph.input_shape:
        input_elements *= dim
    try:
        input_bytes = input_elements * dtype_bytes(graph.precision)
    except KeyError:
        return unknown(source, f"unknown precision {graph.precision!r}")
    elements[input_name] = input_elements
    nbytes[input_name] = input_bytes

    total_elements = 0
    total_bytes = 0.0
    for ly in graph.layers:
        try:
            width = dtype_bytes(ly.act_dtype)
        except KeyError:
            return unknown(source, f"{ly.name}: unknown activation dtype {ly.act_dtype!r}")
        elements[ly.name] = ly.out_elements
        nbytes[ly.name] = ly.out_elements * width
        total_elements += ly.out_elements
        total_bytes += ly.out_elements * width

    last_read: dict[str, int | None] = {name: None for name in elements}
    for index, ly in enumerate(graph.layers):
        if ly.reads:
            consumed = ly.reads
        elif index == 0:
            consumed = (input_name,)
        else:
            consumed = (graph.layers[index - 1].name,)
        for name in consumed:
            if name not in elements:
                return unknown(source, f"{ly.name}: reads {name!r}, which is not an earlier layer")
            last_read[name] = index

    live: set[str] = set()
    if last_read[input_name] is not None:
        live.add(input_name)

    peak_bytes = -1.0
    peak_elements = 0
    peak_at: str | None = None
    for index, ly in enumerate(graph.layers):
        live.add(ly.name)
        live_bytes = sum(nbytes[name] for name in live)
        live_elements = sum(elements[name] for name in live)
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_elements = live_elements
            peak_at = ly.name
        for name in [name for name in live if last_read[name] == index]:
            live.remove(name)

    if peak_at is None:
        peak_bytes = 0.0

    return computed(
        _whole(peak_bytes),
        source,
        peak_at=peak_at,
        peak_elements=peak_elements,
        total_elements=total_elements,
        total_bytes=float(total_bytes),
        includes_network_input=True,
        note="peak is the resident set, not the largest single tensor",
    )

# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """
    if convention not in FLOP_CONVENTIONS:
        return unknown("to_flops", f"unrecognised convention {convention!r}")

    scale = FLOPS_PER_MAC if convention == "mac_is_two_flops" else FLOP_CONVENTIONS[convention]
    if not isinstance(macs, dict) or not is_answered(macs) or macs.get("value") is None:
        source = macs.get("source", "to_flops") if isinstance(macs, dict) else "to_flops"
        detail = "MAC count is unknown"
        if isinstance(macs, dict) and macs.get("detail"):
            detail = str(macs["detail"])
        finding = unknown(source, detail)
        finding["convention"] = convention
        return finding

    fields: dict[str, Any] = {
        "convention": convention,
        "flops_per_mac": scale,
    }
    if isinstance(macs.get("per_layer"), dict):
        fields["per_layer"] = {
            name: _whole(value * scale) for name, value in macs["per_layer"].items()
        }
    fields["note"] = "a count of operations contains no unit of time"

    return computed(
        _whole(macs["value"] * scale),
        str(macs.get("source", "to_flops")),
        **fields,
    )
