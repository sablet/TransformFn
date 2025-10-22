from __future__ import annotations

from typing import TypedDict

from algo_trade_dtypes.generators import HLOCVSpec
from algo_trade_dtypes.registry import HLOCVSpecReg
from xform_core import RegisteredType, TransformFn, transform


class SpecWrapper(TypedDict):
    spec: HLOCVSpec


def check_spec_wrapper(wrapper: SpecWrapper) -> None:
    if "spec" not in wrapper:
        raise ValueError("wrapper must include spec")


def test_auto_annotation_uses_registered_examples() -> None:
    # Register HLOCVSpec examples and DataFrame checks for this test
    HLOCVSpecReg.register()
    wrapper_reg = (
        RegisteredType(SpecWrapper)
        .with_example({"spec": HLOCVSpec(n=8, seed=1)}, "spec_wrapper_default")
        .with_check(check_spec_wrapper)
    )
    wrapper_reg.register()

    def generate(spec: HLOCVSpec) -> SpecWrapper:
        """Auto-annotation should pull ExampleValue from registry."""

        return {"spec": spec}

    decorated = transform(generate)
    transform_fn = decorated.__transform_fn__
    assert isinstance(transform_fn, TransformFn)
    assert transform_fn.input_metadata
    assert transform_fn.output_checks
