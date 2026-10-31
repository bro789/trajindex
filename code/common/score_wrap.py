"""Run score_feasibility.py on machine 6, where the openai-agents SDK is not installed.
The harness only imports it at module level for the agent loop; deterministic scoring
never touches it, so empty placeholder modules are enough."""
import runpy, sys, types
class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})
for m in ("agents", "agents.editor"):
    sys.modules[m] = _Stub(m)
runpy.run_path(__file__.replace("score_wrap.py", "score_feasibility.py"), run_name="__main__")

