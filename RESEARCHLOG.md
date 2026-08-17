# RESEARCHLOG

Append-only Log für überraschende Ergebnisse, Bugs mit nicht-offensichtlicher Ursache und verworfene Ansätze.

## 2026-07-13 — Job 1881087 (GLM-5.2-FP8, 4 Nodes): Hang durch transienten DeepGEMM-JIT-Compile-Fehler

**Symptom:** vLLM kam nie hoch; ab 08:33 nur noch endlos `No available shared memory broadcast block found in 60 seconds` (EngineCore wartet auf tote Worker). Job lief als Zombie weiter und verbrannte GPU-Stunden.

**Ursache:** Während des DeepGEMM-Warmups (~2000 JIT-Kernel pro Rank, 16 Ranks) schlug auf n25g0010 bei genau 2 von 4 Worker-Prozessen (PP1_TP0/TP1) die NVCC-Kompilierung fehl:
`error: missing binary operator before token "(" bei _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()` in den CCCL-Headern von CUDA 13.0.2 (cvmfs). Die Worker starben, vLLM (Ray-Backend) hängte sich auf statt abzubrechen.

**Nicht-offensichtlich:**
- 89 Kernel desselben Typs mit *identischem Include-Hash* kompilierten auf demselben Node erfolgreich; nur 2 schlugen fehl.
- Der exakt gleiche fehlgeschlagene `kernel.cu` kompilierte später auf demselben Node mit demselben nvcc-Befehl (aus `_C.so` extrahierte Original-Flags, `env -i`) fehlerfrei → **transienter Fehler**, kein deterministisches Inkompatibilitätsproblem. Verdacht: cvmfs-Aussetzer unter paralleler JIT-Last (16 Ranks lesen gleichzeitig CUDA-Header von cvmfs) → unvollständig gelieferter Config-Header → Makro undefiniert.
- Achtung bei manueller Reproduktion: mit Login-Shell-Umgebung (geladene Module, CPATH) schlägt der Compile *deterministisch* mit derselben Meldung fehl (CCCL-Header-Mischung). Für aussagekräftige Tests `env -i` verwenden.

**Konsequenzen / Ideen:**
- vLLM-Hang statt Fail-Fast: Watchdog im sbatch sinnvoll (wenn `/v1/models` nach N min nicht erreichbar → Job abbrechen).
- JIT-Rekompilierlast senken: DeepGEMM-Cache (`DG_JIT_CACHE_DIR`) nach erfolgreichem Warmup nach $HPCWORK sichern und beim Jobstart auf die node-lokalen NVMe-Caches vorab kopieren (Kopieren statt geteiltem Cache → kein NFS-Locking-Problem wie früher mit "Stale file handle").

## 2026-07-13 — Job 1881487: Root Cause der wiederkehrenden Worker-Abstürze = SIGBUS durch mmap-Page-Fault auf NFS (venv auf $HPCWORK)

**Symptom:** Server lief 50 min stabil und beantwortete Requests, dann starb Ray-Worker PP1_TP2 (PID 332547, n25g0012) um 09:46:52 "unexpectedly" → EngineCore fatal → ganzer vLLM-Server tot. Kein OOM: MEMMON zeigte durchgehend RAM used=3 %, /dev/shm 523 MB von 378 G. Die frühere Hypothese "langsamer RAM-Leak → OOM/SIGBUS nach Stunden" ist damit **widerlegt**.

**Root-Cause-Analyse (Core-Dump-Forensik):**
- `core_pattern=core.%h.%p.%s` → Suffix `.7` der Core-Files = **Signal 7 = SIGBUS**. Im Repo liegen mehrere alte `.7`-Cores (24./25.06., überwiegend n25g0012) → derselbe Absturz trat schon mehrfach auf.
- Log: `*** SIGBUS received ***`, direkt gefolgt von `symbolize_elf.inc: read failed: errno=5` und `libnccl.so.2: wrong elf type: -1` → schon der Crash-Handler konnte die .so **nicht mehr vom Dateisystem lesen (EIO)**.
- Core-Dump (NT_FILE + PRSTATUS): Absturz-PC `0x1444a8367c92` liegt im gemappten Text-Segment von `.venv/.../nvidia/nccl/lib/libnccl.so.2` — das venv liegt auf `$HPCWORK` (NFS). Prozesszustand beim Dump: `D` (uninterruptible I/O-Wait).
- **Mechanismus:** Alle ~1045 gemappten Dateien (python3.13-Binary + libpython aus NFS-Home, hunderte .so aus dem venv auf hpcwork-NFS) sind demand-paged. Der Kernel darf saubere Code-Pages jederzeit verwerfen; beim nächsten Ausführen der Stelle wird die Page vom NFS nachgeladen. Schlägt das transient fehl (EIO — gleiche Fehlerklasse wie der cvmfs-Aussetzer von Job 1881087), liefert der Kernel **SIGBUS** an den Prozess. Ein Rank stirbt → vLLM reißt alles ab. Mit 16 Ranks × Stunden Laufzeit ist die Trefferwahrscheinlichkeit hoch.

**Nicht-offensichtlich:** Der Absturz korreliert mit gar nichts im Workload (kam ~20 s NACH Ende des letzten Requests) — weil die Ursache außerhalb des Prozesses liegt (NFS-Client/Server-Hiccup beim Page-in). Deshalb sahen die Crashes "zufällig" aus.

**Fix (`runGLMStable.sbatch`):** venv (9,2 G) + uv-CPython-Distribution (83 M) werden beim Jobstart per srun auf **jedem** Node nach `$TMPDIR` (node-lokales NVMe `/w0`, jobweit identischer Pfad) kopiert; `pyvenv.cfg`-`home`, `bin/python`-Symlink und Entry-Point-Shebangs werden auf die lokale Kopie umgebogen. Ray & vLLM laufen komplett vom lokalen NVMe → kein Page-in über NFS mehr. Zusätzlich: Startup-Watchdog + Health-Monitor mit begrenzten Neustarts (Konsequenz aus Job 1881087). Restrisiko dokumentiert: wenige kleine cvmfs-Libs (libstdc++ u. a., lokal gecacht) und HF-Weights-mmap nur während der Ladephase.
