import pytest

from tilebench.cubin import count_opcodes, parse_resources, parse_threads

# Captured from cuobjdump on real kernels.
RESOURCES = """
Resource usage:
 Common:
  GLOBAL:64 CONSTANT[4]:8
 Function whole_pointer:
  REG:255 STACK:400 SHARED:16 LOCAL:0 CONSTANT[0]:396 TEXTURE:0 SURFACE:0 SAMPLER:0
"""

ELF = """
	Attribute:	EIATTR_MAXREG_COUNT
	Format:	EIFMT_HVAL
	Value:	0xff
	Attribute:	EIATTR_REQNTID
	Format:	EIFMT_SVAL
	Value:	0x80 0x1 0x1
"""

SASS = """
		/*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;   /* 0x00000a00ff017624 */
                                                                             /* 0x000fc400078e00ff */
		/*0010*/               @P0 LDG.E.U16 R2, [R4.64] ;
		/*0020*/              @!P1 LDG.E.128 R8, [R6.64] ;
		/*0030*/                   LDGSTS.E.BYPASS.128 [R3], [R4.64] ;
		/*0040*/                   STL.64 [R1+0x8], R2 ;
		/*0050*/                   LDL.LU R2, [R1+0x8] ;
		/*0060*/                   BAR.SYNC 0x0 ;
		/*0070*/                   NOP ;
"""


def test_resources_come_from_the_kernel_not_the_common_section() -> None:
    assert parse_resources(RESOURCES) == ("whole_pointer", 255, 400, 16)


def test_several_kernels_are_rejected_rather_than_misreported() -> None:
    with pytest.raises(ValueError, match="found 2"):
        parse_resources(RESOURCES + RESOURCES)


def test_threads_multiply_all_three_dimensions() -> None:
    assert parse_threads(ELF) == 128
    assert parse_threads(ELF.replace("0x80 0x1 0x1", "0x20 0x4 0x2")) == 256


def test_a_cubin_without_a_fixed_block_size_is_an_error() -> None:
    with pytest.raises(ValueError, match="block size"):
        parse_threads("Attribute:	EIATTR_MAXREG_COUNT")


def test_opcodes_ignore_predicates_and_modifiers() -> None:
    opcodes = count_opcodes(SASS)
    assert opcodes["LDG"] == 2
    assert opcodes["LDGSTS"] == 1
    assert opcodes["LDL"] == 1
    assert opcodes["STL"] == 1
    assert opcodes["BAR"] == 1


def test_hex_encoding_lines_are_not_counted_as_instructions() -> None:
    assert count_opcodes(SASS).total() == 8
