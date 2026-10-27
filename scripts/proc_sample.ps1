param([int]$ProcessId)

# Print "<working-set bytes> <thread count>" for a process AND all of its
# descendants.
#
# Why this file exists: on this machine venv\Scripts\python.exe is a launcher
# stub that spawns a child interpreter which does the actual work. Measuring
# only the PID returned by Popen reports the stub (a few MB) and misses the
# real process (200+ MB with torch loaded), which makes any "memory is
# stable / leaking" verdict meaningless. Walking the tree fixes that.

$ErrorActionPreference = "SilentlyContinue"
$totalWs = 0
$totalTh = 0

function Measure-Tree([int]$Id, [int]$Depth) {
    # Depth guard: a broken parent chain must not recurse forever.
    if ($Depth -gt 8) { return }

    $proc = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if ($proc) {
        $script:totalWs += $proc.WorkingSet64
        $script:totalTh += $proc.Threads.Count
    }

    $children = Get-CimInstance Win32_Process `
        -Filter ("ParentProcessId=" + $Id) -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Measure-Tree $child.ProcessId ($Depth + 1)
    }
}

Measure-Tree $ProcessId 0
Write-Output ($script:totalWs.ToString() + " " + $script:totalTh.ToString())
