"""Row-scaled FP8 with FP32 scales: layout, codec, writer path and compressed source."""

from __future__ import annotations

import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.fp8_row import (
    decode_fp8_row_fp32_words,
    dequantize_fp8_row_fp32,
    encode_fp8_row_fp32,
    encode_fp8_row_scaled,
    validate_fp8_row_fp32_words,
)
from tools.artifact.formats import (
    FP8_E4M3FN_ROW_BF16,
    FP8_E4M3FN_ROW_FP32,
    FP8_ROW_FORMATS,
    get_format,
)
from tools.artifact.inspect import artifact_summary
from tools.artifact.layouts import encoded_size, row_scale_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.sources.compressed_tensors import (
    compressed_matrix_source,
    matrix_source,
)
from tools.convert.sources.safetensors import SafetensorsSource

FP32 = "fp8_e4m3fn_row_fp32"
BF16 = "fp8_e4m3fn_row_bf16"


def _f32(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_format_registry_and_scale_geometry():
    assert get_format(FP32) is FP8_E4M3FN_ROW_FP32
    assert FP8_E4M3FN_ROW_FP32.scale_dtype == "fp32"
    assert FP8_E4M3FN_ROW_BF16.scale_dtype == "bf16"
    assert set(FP8_ROW_FORMATS) == {BF16, FP32}
    wide = row_scale_geometry(FP32, (2, 4))
    assert (
        wide.code_plane_bytes,
        wide.scale_plane_offset,
        wide.scale_plane_bytes,
        wide.payload_bytes,
    ) == (8, 256, 8, 264)
    narrow = row_scale_geometry(BF16, (2, 4))
    assert (narrow.scale_plane_offset, narrow.scale_plane_bytes) == (256, 4)
    assert encoded_size("row_scale_v1", FP32, (2, 4)) == 264
    # The code plane pads to 256 bytes independent of the scale width.
    large = row_scale_geometry(FP32, (3, 300))
    assert (large.scale_plane_offset, large.payload_bytes) == (1024, 1024 + 12)


def test_known_words_padding_and_exact_reconstruction():
    shape = (2, 4)
    codes = torch.tensor([[0x00, 0x80, 0x38, 0xB8], [0x40, 0xC0, 0x7E, 0xFE]],
                         dtype=torch.uint8)
    scales = _f32(0.5, 2.0)
    payload = encode_fp8_row_fp32(codes, scales, shape)
    assert len(payload) == 264
    assert payload[:8] == bytes(codes.reshape(-1).tolist())
    assert payload[8:256] == bytes(248)
    assert payload[256:] == struct.pack("<ff", 0.5, 2.0)
    decoded_codes, decoded_scales = decode_fp8_row_fp32_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales.view(torch.int32), scales.view(torch.int32))
    assert torch.equal(
        dequantize_fp8_row_fp32(payload, shape),
        torch.tensor([[0.0, -0.0, 0.5, -0.5], [4.0, -4.0, 896.0, -896.0]]),
    )


def test_scale_bits_are_not_rounded_to_bf16():
    # 1 + 2**-20 is representable in FP32 but not in BF16.
    scale = struct.unpack("<f", struct.pack("<I", 0x3F800008))[0]
    codes = torch.tensor([[0x38]], dtype=torch.uint8)
    payload = encode_fp8_row_fp32(codes, _f32(scale), (1, 1))
    assert payload[256:] == struct.pack("<I", 0x3F800008)
    value = dequantize_fp8_row_fp32(payload, (1, 1)).item()
    assert value == struct.unpack("<f", struct.pack("<I", 0x3F800008))[0]
    with pytest.raises(TypeError, match="FP32"):
        encode_fp8_row_fp32(codes, _f32(scale).to(torch.bfloat16), (1, 1))
    with pytest.raises(TypeError, match="BF16"):
        encode_fp8_row_scaled(codes, _f32(scale), (1, 1))


@pytest.mark.parametrize(
    "codes,scales,message",
    [
        ([[0x7F, 0x00]], [1.0], "finite E4M3FN"),
        ([[0xFF, 0x00]], [1.0], "finite E4M3FN"),
        ([[0x00, 0x00]], [-1.0], "nonnegative finite FP32"),
        ([[0x00, 0x00]], [float("nan")], "nonnegative finite FP32"),
        ([[0x00, 0x00]], [float("inf")], "nonnegative finite FP32"),
        ([[0x38, 0x00]], [0.0], "zero row scale"),
    ],
)
def test_fp32_row_words_reject_invalid_input(codes, scales, message):
    codes = torch.tensor(codes, dtype=torch.uint8)
    with pytest.raises(ValueError, match=message):
        encode_fp8_row_fp32(codes, _f32(*scales), (1, 2))
    with pytest.raises(ValueError, match=message):
        validate_fp8_row_fp32_words(codes, _f32(*scales))


def test_fp32_row_codec_rejects_shape_and_payload_mismatch():
    codes = torch.zeros((2, 2), dtype=torch.uint8)
    with pytest.raises(TypeError):
        encode_fp8_row_fp32(codes, _f32(1.0), (2, 2))
    with pytest.raises(TypeError):
        encode_fp8_row_fp32(codes.to(torch.int16), _f32(1.0, 1.0), (2, 2))
    with pytest.raises(ValueError, match="payload"):
        decode_fp8_row_fp32_words(bytes(10), (2, 2))


def test_writer_places_chunked_rows_at_four_byte_scale_stride(tmp_path):
    n, k = 5, 3
    gen = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 0x7E, (n, k), generator=gen).to(torch.uint8)
    scales = torch.rand(n, generator=gen) + 0.25
    path = tmp_path / "row.ninfer"
    specs = [
        ResourceSpec("resource/blob", 4),
        TensorSpec("w", (n, k), FP32, "row_scale_v1"),
    ]
    with ArtifactWriter(
        path, specs, components={"text": {"config": {}}}, bindings={}
    ) as writer:
        writer.write_object("resource/blob", b"abcd")
        output = TensorOutput(writer, "w")
        for begin, end in ((0, 2), (2, 3), (3, 5)):
            output.write_codes(begin, codes[begin:end], scales[begin:end])
    with Artifact(path) as artifact:
        payload = artifact.read_object("w")
        summary = artifact_summary(artifact)
    assert payload == encode_fp8_row_fp32(codes, scales, (n, k))
    decoded_codes, decoded_scales = decode_fp8_row_fp32_words(payload, (n, k))
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)
    assert summary["formats"] == {FP32: 1}
    assert summary["layouts"] == {"row_scale_v1": 1}


def test_writer_rejects_wrong_scale_dtype_for_the_format(tmp_path):
    codes = torch.zeros((1, 2), dtype=torch.uint8)
    for fmt, scales in ((FP32, _f32(1.0).to(torch.bfloat16)), (BF16, _f32(1.0))):
        path = tmp_path / f"{fmt}.ninfer"
        with pytest.raises(TypeError):
            with ArtifactWriter(
                path,
                [TensorSpec("w", (1, 2), fmt, "row_scale_v1")],
                components={"text": {"config": {}}},
                bindings={},
            ) as writer:
                TensorOutput(writer, "w").write_codes(0, codes, scales)
        assert not path.exists()


CODES = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)


def _checkpoint(tmp_path, scales):
    tensors = {
        "proj.weight": CODES.view(torch.float8_e4m3fn),
        "proj.weight_scale": scales,
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return SafetensorsSource(tmp_path)


@pytest.mark.parametrize("shape", [(2, 1), (2,)])
def test_compressed_source_accepts_f32_scales_of_both_shapes(tmp_path, shape):
    scales = _f32(2.0, 0.5).reshape(shape)
    with _checkpoint(tmp_path, scales) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), FP32)
        assert source.encoding() == FP32
        words = source.read_encoded(0, 2)
        assert words.format == FP32 and words.scales.dtype == torch.float32
        assert torch.equal(words.codes, CODES)
        assert torch.equal(words.scales, _f32(2.0, 0.5))
        assert torch.equal(
            source.values().reshape(2, 3),
            torch.tensor([[2.0, -2.0, 4.0], [0.25, -0.25, -0.0]]),
        )
        # Automatic detection selects the format from the stored scale dtype.
        auto = matrix_source(store, "proj.weight", (2, 3))
        assert auto.encoding() == FP32


def test_compressed_source_keeps_bf16_scales_on_the_bf16_format(tmp_path):
    scales = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    with _checkpoint(tmp_path, scales) as store:
        assert matrix_source(store, "proj.weight", (2, 3)).encoding() == BF16
        with pytest.raises(ValueError, match=r"proj\.weight_scale"):
            compressed_matrix_source(store, "proj", (2, 3), FP32).read_encoded(0, 2)


def test_f32_scales_are_never_rounded_to_bf16(tmp_path):
    with _checkpoint(tmp_path, _f32(2.0, 0.5).reshape(2, 1)) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), BF16)
        with pytest.raises(ValueError, match=r"proj\.weight_scale.*F32"):
            source.read_encoded(0, 2)
        with pytest.raises(ValueError, match=r"proj\.weight_scale.*F32"):
            source.values()


@pytest.mark.parametrize(
    "scales,message",
    [
        (_f32(1.0, 1.0, 1.0), r"proj\.weight_scale"),
        (_f32(1.0, 1.0, 1.0, 1.0).reshape(2, 2), r"proj\.weight_scale"),
        (_f32(-1.0, 1.0).reshape(2, 1), "nonnegative finite FP32"),
        (_f32(float("nan"), 1.0).reshape(2, 1), "nonnegative finite FP32"),
        (_f32(0.0, 1.0).reshape(2, 1), "zero row scale"),
    ],
)
def test_f32_scale_validation_names_the_parameter(tmp_path, scales, message):
    with _checkpoint(tmp_path, scales) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), FP32)
        with pytest.raises(ValueError, match=message):
            source.read_encoded(0, 2)


def test_unknown_encoded_format_is_rejected(tmp_path):
    with _checkpoint(tmp_path, _f32(1.0, 1.0)) as store:
        with pytest.raises(ValueError, match="unsupported encoded source format"):
            compressed_matrix_source(store, "proj", (2, 3), "fp8_e4m3fn_row_fp16")
