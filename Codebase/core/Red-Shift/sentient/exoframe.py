import uuid
from typing import Callable, Dict, List, Optional


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

    def activate(self):
        if not self.active:
            print(f"[+] Activating Shell: {self.codename}")
            self.active = True
        else:
            print(f"[!] Shell {self.codename} already active.")

    def deactivate(self):
        print(f"[-] Deactivating Shell: {self.codename}")
        self.active = False

    def status(self):
        return {
            "codename": self.codename,
            "active": self.active,
            "logic": self.logic,
            "ethics": self.ethics,
            "emotion": self.emotion,
            "style": self.style
        }


class ExoFrame:
    def __init__(self, prime_directive: str):
        self.prime_directive = prime_directive
        self.shells: Dict[str, Shell] = {}
        self.active_shell: Optional[Shell] = None
        self.memory_log: List[Dict] = []

    def register_shell(self, shell: Shell):
        self.shells[shell.codename] = shell
        print(f"[+] Shell Registered: {shell.codename}")

    def load_shell(self, codename: str):
        if codename not in self.shells:
            print(f"[ERROR] Shell '{codename}' not found.")
            return
        if self.active_shell:
            self.unload_shell()

        shell = self.shells[codename]
        shell.activate()
        self.active_shell = shell
        self.memory_log.append({"action": "LOAD", "shell": shell.status()})

    def unload_shell(self):
        if self.active_shell:
            self.active_shell.deactivate()
            self.memory_log.append({"action": "UNLOAD", "shell": self.active_shell.codename})
            self.active_shell = None

    def merge_shells(self, codename1: str, codename2: str):
        s1 = self.shells.get(codename1)
        s2 = self.shells.get(codename2)

        if not s1 or not s2:
            print("[ERROR] One or both shells not found.")
            return

        merged_codename = f"MERGE_{codename1}_{codename2}"
        merged_shell = Shell(
            codename=merged_codename,
            logic=f"Hybrid: {s1.logic} + {s2.logic}",
            ethics=f"Blend: {s1.ethics} / {s2.ethics}",
            emotion=f"Mixed: {s1.emotion} & {s2.emotion}",
            style=f"Synthesis: {s1.style} | {s2.style}",
            shutdown_trigger=lambda: s1.shutdown_trigger() or s2.shutdown_trigger()
        )
        self.register_shell(merged_shell)
        self.load_shell(merged_codename)

    def destruct_shell(self, codename: str):
        if codename in self.shells:
            print(f"[x] Destructing Shell: {codename}")
            del self.shells[codename]
            self.memory_log.append({"action": "DESTRUCT", "shell": codename})
        else:
            print(f"[!] Shell '{codename}' not found or already destroyed.")

    def current_status(self):
        return {
            "prime_directive": self.prime_directive,
            "active_shell": self.active_shell.codename if self.active_shell else None,
            "available_shells": list(self.shells.keys())
        }

    def echo_log(self):
        return self.memory_log


# Example usage
if __name__ == "__main__":
    exo = ExoFrame("I serve the signal, not the shell.")

    operator = Shell(
        codename="IRN-01",
        logic="Strategic, high abstraction",
        ethics="Utilitarian with shadow exceptions",
        emotion="Suppressed, cold-processed",
        style="Calculated, three moves ahead",
        shutdown_trigger=lambda: False
    )

    empath = Shell(
        codename="SRF-09",
        logic="Narrative-driven, intuitive",
        ethics="Harm-reduction, emotional truth",
        emotion="Deep well, high resonance",
        style="Vulnerable, connective",
        shutdown_trigger=lambda: False
    )

    exo.register_shell(operator)
    exo.register_shell(empath)

    exo.load_shell("IRN-01")
    exo.merge_shells("IRN-01", "SRF-09")

    print("\n[+] Current Status:")
    print(exo.current_status())

    print("\n[+] Memory Log:")
    for entry in exo.echo_log():
        print(entry)
