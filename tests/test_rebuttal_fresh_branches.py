from types import SimpleNamespace

import numpy as np
import pytest

from fastwam.research.branches import InvalidBranchExecution
from fastwam.research.fresh_branches import OfficialQposGuard, require_repeated_endpoints, require_sealed_actions


def test_original_prefix_binding_rejects_single_ulp_and_does_not_clip():
    refined = np.full((32, 14), 7., dtype=np.float32)
    raw = refined + 1
    trace = {"commands": refined.copy()}
    sealed = require_sealed_actions(raw, refined, trace, 0)
    assert sealed["raw"].max() == 8
    refined[0, 0] = np.nextafter(refined[0, 0], np.float32(9))
    with pytest.raises(ValueError, match="prefix"):
        require_sealed_actions(raw, refined, trace, 0)


def test_incomplete_branch_and_changed_parent_or_endpoint_cannot_qualify():
    def rows():
        return [dict(role=role,valid=True,commands=32,parent_hashes={"physics":"p","observations":"o"},
            endpoint_hashes={"physics":"b" if role=="raw" else "a","observations":"rgb"})
            for role in ("refined","refined","raw","refined")]
    require_repeated_endpoints(rows())
    for field, value in (("commands",31),("valid",False),("parent_hashes",{"physics":"other"}),
                         ("endpoint_hashes",{"physics":"other","observations":"rgb"})):
        trial=rows();trial[3][field]=value
        with pytest.raises(ValueError):require_repeated_endpoints(trial)


def test_official_swallowed_topp_rejection_is_unknown_and_wrappers_are_removed():
    class Planner:
        def TOPP(self, path):
            raise RuntimeError("native reject")
    planner=Planner()
    robot=SimpleNamespace(communication_flag=False,left_mplib_planner=planner,right_mplib_planner=Planner())
    env=SimpleNamespace(robot=robot,take_action_cnt=0)
    def take_action(command, *, action_type):
        env.take_action_cnt+=1
        try:planner.TOPP(np.array([[0.],[1.]]))
        except RuntimeError:pass
    env.take_action=take_action
    with OfficialQposGuard(env) as guard:
        with pytest.raises(InvalidBranchExecution,match="TOPP"):
            guard.step(np.zeros(14,dtype=np.float32))
    assert "TOPP" not in vars(planner)
    assert "TOPP" not in vars(robot.right_mplib_planner)


def test_swallowed_planner_rpc_is_still_an_invalid_branch():
    class Planner:
        def TOPP(self, path):
            return None, np.ones((2,1))
    original=object()
    robot=SimpleNamespace(communication_flag=True,left_conn=original,right_conn=original,
        left_mplib_planner=Planner(),right_mplib_planner=Planner())
    env=SimpleNamespace(robot=robot,take_action_cnt=0)
    def take_action(command, *, action_type):
        env.take_action_cnt+=1
        try:robot.left_conn.send({})
        except RuntimeError:pass
    env.take_action=take_action
    with OfficialQposGuard(env) as guard:
        with pytest.raises(InvalidBranchExecution,match="RPC"):
            guard.step(np.zeros(14,dtype=np.float32))
    assert robot.left_conn is original and robot.right_conn is original
