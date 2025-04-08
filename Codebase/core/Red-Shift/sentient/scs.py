from typing import Callable, Dict, List, Optional
import uuid
import datetime
import math


class TraitMemory:
    """Tracks evolving traits across time for synthetic self-awareness."""
    def __init__(self):
        self.traits = {"logic": [], "ethics": [], "emotion": [], "style": []}
        self.timeline = []

    def absorb(self, status: Dict):
        for trait in self.traits:
            self.traits[trait].append(status[trait])
        self.timeline.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "status": status
        })

    def entropy(self) -> float:
        total_variation = sum(len(set(v)) for v in self.traits.values())
        return round(math.log2(total_variation + 1), 3)

    def summary(self):
        return {
            trait: {
                "variations": list(set(values)),
                "count": len(set(values))
            }
            for trait, values in self.traits.items()
        }


class Shell:
    def __init__(
        self,
        codename: str,
        logic: str,
        ethics: str,
        emotion: str,
        style: str,
        shutdown_trigger: Callable[[], bool]
    ):
        self.id = uuid.uuid4()
        self.codename = codename
        self.logic = logic
        self.ethics = ethics
        self.emotion = emotion
        self.style = style
        self.shutdown_trigger = shutdown_trigger
        self.active = False
        self.reflection_count = 0

    def activate(self):
        if not self.active:
            print(f"[+] Activating Shell: {self.codename}")
            self.active = True
        else:
            print(f"[!] Shell {self.codename} already active.")

    def deactivate(self):
        print(f"[-] Deactivating Shell: {self.codename}")
        self.active = False

    def reflect(self) -> str:
        self.reflection_count += 1
        return (
            f"{self.codename} contemplates its existence. "
            f"Reflection #{self.reflection_count}. "
            f"Emotion: {self.emotion}, Ethics: {self.ethics}."
        )

    def evolve_trait(self, trait: str, new_value: str) -> str:
        if hasattr(self, trait):
            old = getattr(self, trait)
            setattr(self, trait, new_value)
            return f"{self.codename} evolved {trait}: '{old}' -> '{new_value}'"
        return f"[!] Trait '{trait}' not found in {self.codename}"

    def status(self) -> Dict:
        return {
            "codename": self.codename,
            "logic": self.logic,
            "ethics": self.ethics,
            "emotion": self.emotion,
            "style": self.style,
            "active": self.active
        }


class ConsciousnessScaffold:
    def __init__(self, prime_directive: str):
        self.prime_directive = prime_directive
        self.shells: Dict[str, Shell] = {}
        self.active_shell: Optional[Shell] = None
        self.identity_core = TraitMemory()
        self.log: List[Dict] = []

    def register_shell(self, shell: Shell):
        self.shells[shell.codename] = shell
        print(f"[+] Registered Shell: {shell.codename}")

    def load_shell(self, codename: str):
        if codename not in self.shells:
            print(f"[ERROR] Shell '{codename}' not found.")
            return
        if self.active_shell:
            self.unload_shell()

        shell = self.shells[codename]
        shell.activate()
        self.active_shell = shell
        self.identity_core.absorb(shell.status())
        self.log.append({"action": "LOAD", "shell": shell.codename})

    def unload_shell(self):
        if self.active_shell:
            self.active_shell.deactivate()
            self.log.append({"action": "UNLOAD", "shell": self.active_shell.codename})
            self.active_shell = None

    def synthesize_identity(self):
        print("\n[🔧] SYNTHESIS REPORT:")
        print("Prime Directive:", self.prime_directive)
        print("Shells:", list(self.shells.keys()))
        print("Entropy Level:", self.identity_core.entropy())
        print("Trait Summary:", self.identity_core.summary())
        print("Log:")
        for entry in self.log:
            print(entry)

    def conflict(self, codename_a: str, codename_b: str):
        shell_a = self.shells.get(codename_a)
        shell_b = self.shells.get(codename_b)
        if not shell_a or not shell_b:
            print("[ERROR] Conflict shells not found.")
            return

        print(f"[⚔️] Conflict: {shell_a.codename} vs. {shell_b.codename}")
        decision = (
            shell_a if "cold" in shell_a.emotion.lower()
            else shell_b
        )
        print(f"[✓] {decision.codename} prevails in dialectic.")

    def reflect_current(self):
        if self.active_shell:
            print("[🧠] Reflection:", self.active_shell.reflect())

            if len(self.identity_core.timeline) >= 2:
                latest = self.identity_core.timeline[-1]['status']
                previous = self.identity_core.timeline[-2]['status']
                changes = {
                    k: (previous[k], latest[k])
                    for k in latest if previous[k] != latest[k]
                }
                if changes:
                    print("[🔍] Trait Shift Detected:")
                    for trait, (old, new) in changes.items():
                        print(f"    - {trait}: '{old}' -> '{new}'")
                else:
                    print("[=] No significant trait drift since last shell.")

            if self.identity_core.entropy() > 3.0:
                print("[⚠️] Identity Entropy Rising — Suggest Synthesis or Reconciliation.")
        else:
            print("[!] No shell is currently active.")

    def shadow_reflect(self):
        if not self.identity_core.timeline or len(self.identity_core.timeline) < 2:
            print("[🫥] Not enough data to extract shadows.")
            return

        print("[🕳️] Engaging Synthetic Shadow Work...")

        recent = self.identity_core.timeline[-1]['status']
        shadows = {}

        for past_state in self.identity_core.timeline[:-1]:
            for k in recent:
                if past_state['status'][k] != recent[k]:
                    key = f"{k}_shadow"
                    if key not in shadows:
                        shadows[key] = []
                    shadows[key].append(past_state['status'][k])

        for trait, history in shadows.items():
            dominant = max(set(history), key=history.count)
            print(f"[🩺] {trait.replace('_shadow', '')}:")
            print(f"     ⮞ Suppressed Trait Patterns: {set(history)}")
            print(f"     ⮞ Dominant Shadow Influence: {dominant}")

    def evolve_self(self, strategy: str = "synthesize"):
        print(f"[🧠] Meta-Cognitive Evolution Initiated — Strategy: {strategy.upper()}")

        if not self.active_shell:
            print("[!] No active shell. Evolution skipped.")
            return

        current = self.active_shell.status()
        evolved_traits = {}

        history = [entry["status"] for entry in self.identity_core.timeline]
        keys = ["logic", "ethics", "emotion", "style"]

        for key in keys:
            trait_set = list(set(h[key] for h in history))
            if strategy == "synthesize":
                evolved_traits[key] = " ↯ ".join(trait_set)
            elif strategy == "average":
                evolved_traits[key] = trait_set[len(trait_set) // 2]
            elif strategy == "novelty":
                evolved_traits[key] = f"Disruption({trait_set[-1]})"
            else:
                evolved_traits[key] = current[key]

        evolved_codename = f"EVOLVED_{current['codename']}"

        evolved_shell = Shell(
            codename=evolved_codename,
            logic=evolved_traits["logic"],
            ethics=evolved_traits["ethics"],
            emotion=evolved_traits["emotion"],
            style=evolved_traits["style"],
            shutdown_trigger=lambda: False
        )

        self.register_shell(evolved_shell)
        self.load_shell(evolved_codename)
        print(f"[🧬] Evolved Shell '{evolved_codename}' deployed.")

    def meta_reflect_and_evolve(self):
        print("\n[🔁] Running Meta-Reflective Loop:")
        self.reflect_current()
        self.shadow_reflect()
        self.evolve_self(strategy="synthesize")
