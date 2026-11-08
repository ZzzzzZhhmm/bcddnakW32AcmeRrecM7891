"""State bookkeeping fixtures do not qualify SAPIEN or task branches."""
from types import SimpleNamespace
import numpy as np
import pytest

from fastwam.research.rmbench_branches import python_state, restore_python, _IdleChannel, exact_hold_path


def test_value_state_is_independent_and_new_caches_are_removed():
    handle = object()
    obj = SimpleNamespace(stage_id=1, cache={"x": np.arange(3)}, handle=handle)
    state = python_state(obj)
    obj.stage_id=4; obj.cache["x"][:]=0; obj.transient=[1,2]
    restore_python(obj,state)
    assert obj.stage_id==1 and np.array_equal(obj.cache["x"],np.arange(3))
    assert not hasattr(obj,"transient") and obj.handle is handle


def test_opaque_topology_change_is_rejected():
    obj = SimpleNamespace(handle=object(), counter=1)
    state = python_state(obj);obj.handle=object()
    with pytest.raises(RuntimeError,match="topology"):
        restore_python(obj,state)


def test_idle_planner_cannot_hide_subprocess_state_changes():
    guard = _IdleChannel(SimpleNamespace(closed=False))
    assert not guard.closed
    with pytest.raises(RuntimeError,match="subprocess state"):
        guard.send({"cmd":"plan_path"})
    with pytest.raises(RuntimeError,match="subprocess state"):
        guard.recv()


@pytest.mark.parametrize("path,expected", [
    ([[0.,0.],[0.,0.]],True),
    ([[0.,0.],[1e-10,0.]],False),
    ([[0.,0.],[.1,0.],[0.,0.]],False),
])
def test_only_an_exact_hold_is_exempt_from_empty_trajectory_failure(path,expected):
    assert exact_hold_path(path) is expected


@pytest.mark.parametrize("mutation",[None,"physics","task","rng"])
def test_query_canonicalization_requires_no_physics_step_or_state_change(monkeypatch,mutation):
    import fastwam.research.rmbench_branches as module
    class Physics:
        def __init__(self):self.value=b"native"
        def pack(self):return self.value
        def unpack(self,value):self.value=b"changed" if mutation=="physics" else value
    class Scene:
        def __init__(self):self.physx_system=Physics()
        def pack_poses(self):return b"pose cache"
        def get_all_articulations(self):return []
        def step(self):raise AssertionError("canonicalization must not step physics")
    class Env:
        def __init__(self):
            self.scene=Scene();self.robot=SimpleNamespace(value=1);self.take_action_cnt=0;self.now_obs={}
        def _update_render(self):pass
        def get_obs(self):
            if mutation=="task":self.take_action_cnt+=1
            self.now_obs={"rgb":np.ones((2,2,3),dtype=np.uint8)}
            return self.now_obs
    calls=[]
    def rng():
        calls.append(1)
        return {"value":len(calls) if mutation=="rng" else 0}
    monkeypatch.setattr(module,"rng_state",rng)
    if mutation:
        with pytest.raises(RuntimeError,match="canonicalization changed"):
            module.canonicalize_query(Env())
    else:
        observation,proof=module.canonicalize_query(Env())
        assert observation["rgb"].shape==(2,2,3)
        assert proof["cpu_before_sha256"]==proof["cpu_after_sha256"] and proof["physics_steps"]==0


@pytest.mark.parametrize("cycle",[False,True])
def test_native_checkpoint_preparation_records_reencoding_and_rejects_cycles(monkeypatch,cycle):
    import fastwam.research.rmbench_branches as module
    class Physics:
        value=b"initial"
        def pack(self):return self.value
        def unpack(self,value):
            self.value=(b"other" if value==b"initial" else b"initial") if cycle else b"stable"
    class Scene:
        def __init__(self):self.physx_system=Physics()
        def pack_poses(self):return self.physx_system.pack()
        def get_all_articulations(self):return []
        def step(self):raise AssertionError("no physical step permitted")
    class Env:
        def __init__(self):self.scene=Scene();self.robot=SimpleNamespace(value=1);self.now_obs={}
        def _update_render(self):pass
        def get_obs(self):self.now_obs={"value":1};return self.now_obs
    monkeypatch.setattr(module,"rng_state",lambda:{"state":1})
    rows=[]
    if cycle:
        with pytest.raises(RuntimeError,match="exact fixed point"):
            module.prepare_native_query(Env(),max_rounds=3,emit=rows.append)
        assert len(rows)==3
    else:
        _,proof=module.prepare_native_query(Env(),max_rounds=3,emit=rows.append)
        assert proof["original_cpu_encoding_changed"] and proof["rounds"]==2
        assert proof["cpu_before_sha256"]==proof["cpu_after_sha256"] and proof["physics_steps"]==0


@pytest.mark.parametrize("rounds",[0,True,6])
def test_invalid_native_round_budget_is_rejected_before_environment_access(rounds):
    from fastwam.research.rmbench_branches import prepare_native_query
    with pytest.raises(ValueError,match="1..5"):
        prepare_native_query(object(),max_rounds=rounds)
