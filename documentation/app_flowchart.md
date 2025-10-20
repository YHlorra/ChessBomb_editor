flowchart TD
    Start[Start Application] --> Resources[Load Resources]
    Resources --> Editor[Interactive Board Editor]
    Editor --> ConfigComplete{Board Configuration Complete}
    ConfigComplete -- No --> Editor
    ConfigComplete -- Yes --> Solve[Start Solving]
    Solve --> Thread[Launch Solver Thread]
    Thread --> Beam[Beam Search Solver]
    Beam --> Solved{Solution Found}
    Solved -- Yes --> Solution[Show Solution Window]
    Solved -- No --> NoSolution[Show No Solution Message]
    Solution --> End[End]
    NoSolution --> End