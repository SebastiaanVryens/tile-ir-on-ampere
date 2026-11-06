from tilebench.schedule import hot_path, loads_in_flight, parse, read_schedule

# The shape of a real layer norm: loads, a barrier, more loads, then a slow path after EXIT.
SASS = """
        /*0000*/                   LDG.E.128 R4, [R2.64] ;   /* 0x0000000402047981 */
                                                             /* 0x000ea8000c1e1d00 */
        /*0010*/               @P0 LDG.E.128 R8, [R2.64+0x10] ;
        /*0020*/                   LDG.E.128 R12, [R2.64+0x20] ;
        /*0030*/                   HADD2.F32 R20, -RZ, R5.reuse.H1_H1 ;
        /*0040*/                   LDG.E.128 R16, [R2.64+0x30] ;
        /*0050*/                   BAR.SYNC 0x0 ;
        /*0060*/                   BSSY B0, 0x90 ;
        /*0070*/                   LDG.E.U16 R24, [R2.64+0x40] ;
        /*0080*/                   FADD R21, R24, R20 ;
        /*0090*/                   LDG.E.64 R26, [R2.64+0x50] ;
        /*00a0*/                   LDL R30, [R1+0x8] ;
        /*00b0*/                   EXIT ;
        /*00c0*/                   LDL R31, [R1+0xc] ;
        /*00d0*/                   BSSY B0, 0xe0 ;
        /*00e0*/                   RET.REL.NODEC R20 0x0 ;
"""


def test_parse_reads_the_opcode_past_a_predicate() -> None:
    names = [item.name for item in parse(SASS)]
    assert names[:3] == ["LDG.E.128", "LDG.E.128", "LDG.E.128"]
    assert names[3] == "HADD2.F32"


def test_control_words_are_not_instructions() -> None:
    assert len(parse(SASS)) == 15


def test_hot_path_stops_at_the_last_exit() -> None:
    assert hot_path(parse(SASS))[-1].base == "EXIT"


def test_load_width_sets_the_registers_it_writes() -> None:
    wide, _, _, _, _, _, _, narrow, _, pair = parse(SASS)[:10]
    assert wide.load_target == {4, 5, 6, 7}
    assert narrow.load_target == {24}
    assert pair.load_target == {26, 27}


def test_a_load_stays_in_flight_until_its_register_is_used() -> None:
    # The HADD2 ends the first load only, so the fourth load finds two still waiting.
    assert loads_in_flight(parse(SASS))[0] == [1, 2, 3, 3]


def test_a_use_right_after_a_load_keeps_the_count_low() -> None:
    # The stretch after the barrier also still holds the loads from before it.
    assert loads_in_flight(parse(SASS))[1] == [4, 4]


def test_address_registers_do_not_end_a_load() -> None:
    # The first load writes R4 to R7, and a later load reads its address from R4.
    sass = """
        /*0000*/                   LDG.E.128 R4, [R2.64] ;
        /*0010*/                   LDG.E.128 R8, [R4.64] ;
        /*0020*/                   EXIT ;
    """
    assert loads_in_flight(parse(sass)) == [[1, 1]]


def test_uniform_registers_are_not_general_registers() -> None:
    sass = """
        /*0000*/                   LDG.E.128 R4, [R2.64] ;
        /*0010*/                   UIADD3 UR4, UR5, UR6, URZ ;
        /*0020*/                   LDG.E.128 R8, [R2.64+0x10] ;
        /*0030*/                   EXIT ;
    """
    assert loads_in_flight(parse(sass)) == [[1, 2]]


def test_schedule_leaves_out_the_slow_path() -> None:
    schedule = read_schedule(SASS)
    assert schedule.branch_regions == 1
    assert schedule.spills == 1


def test_schedule_reports_the_first_and_last_stretch() -> None:
    schedule = read_schedule(SASS)
    assert schedule.first_peak == 3
    assert schedule.last_peak == 4


def test_a_kernel_without_loads_has_no_peak() -> None:
    schedule = read_schedule("        /*0000*/                   EXIT ;\n")
    assert (schedule.first_peak, schedule.last_peak) == (0, 0)
