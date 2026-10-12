```mermaid
graph TD
    %% Einstiegspunkte
    subgraph Einstiegspunkte ["Hauptskripte (User Scripts)"]
        Main_TwoMass["main_TwoMass.py"]
        Main_NSP["main_NSP.py"]
        Main_SCARA["main_SCARA.py"]
        Main_FiveBar["main_FiveBarMechanism.py"]
    end

    %% Setup & Toolbox Schicht
    subgraph Setup ["Setup & API Schicht (_setup.py)"]
        SetupFuncs["OptimCtrl / TimeOptimCtrl / ParamIdent"]
        ToolboxClass["Toolbox (toolbox.py)"]
    end

    %% Kernkomponenten
    subgraph Core ["Optimization Toolbox Core"]
        OptimTask["OptimTask (optimization_task.py)"]
        FreeDyn["FreeDyn Management (Management_FreeDyn.py)"]
        Control["Control / Splines (control_cubSPL_zeroClamped.py)"]
        UserFcts["User Functions (user_fcts.py)"]
        UpdateVars["Update Optim Vars (update_optim_vars_if_changed.py)"]
    end

    %% Mathematischer Unterbau & Solver
    subgraph Solver ["Mathematischer Unterbau & Solver"]
        AdjGrad["Adjoint Gradients (adjGrad_wrt_*.py)"]
        BDFCore["BDF Core (BDF_core.py)"]
        BDFImpl["Dense / Sparse BDF (BDF_*_physical_time.py)"]
        BCCore["Boundary Conditions (consistent_boundary_conditions.py)"]
    end

    %% Verbindungen: Hauptskripte -> Setup
    Main_TwoMass --> SetupFuncs
    Main_NSP --> SetupFuncs
    Main_SCARA --> SetupFuncs
    Main_FiveBar --> SetupFuncs

    %% Setup instanziiert Kernkomponenten
    SetupFuncs -->|erstellt| FreeDyn
    SetupFuncs -->|erstellt| Control
    SetupFuncs -->|erstellt| OptimTask
    SetupFuncs -->|verpackt in| ToolboxClass

    %% Toolbox steuert Kernkomponenten
    ToolboxClass -->|nutzt| OptimTask
    ToolboxClass -->|greift auf zu| FreeDyn
    ToolboxClass -->|greift auf zu| Control

    %% OptimTask Interaktionen
    OptimTask -->|initialisiert| UserFcts
    OptimTask -->|initialisiert| UpdateVars
    OptimTask -->|initialisiert| AdjGrad
    OptimTask -->|ruft auf| FreeDyn
    OptimTask -->|ruft auf| Ctrl_Ref[Control]

    %% UpdateOptimVars Interaktionen (Neu ergänzt)
    UpdateVars -->|aktualisiert| Control
    UpdateVars -->|triggert FreeDyn Simulation| FreeDyn

    %% Adjoint Gradienten & Solver Interaktionen (Neu ergänzt)
    AdjGrad -->|verwendet| BDFCore
    AdjGrad -->|verwendet| BCCore
    AdjGrad -->|wertet aus| UserFcts
    AdjGrad -->|greift auf zu| FreeDyn

    %% BDF Verzweigung
    BDFCore --> BDFImpl
