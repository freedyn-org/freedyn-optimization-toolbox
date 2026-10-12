```mermaid
graph TD
    %% Einstiegspunkte
    subgraph Einstiegspunkte ["Hauptskripte (User Scripts)"]
        Main_TwoMass["main_TwoMass.py"]
        Main_NSP["main_NSP.py"]
        Main_SCARA["main_SCARA.py"]
        Main_FiveBar["main_FiveBarMechanism.py"]
    end

    %% Setup & Toolbox
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
    end

    subgraph Solver ["Mathematischer Unterbau (Adjoint & BDF)"]
        AdjGrad["Adjoint Gradients"]
        BDF["BDF Core & Solvers (BDF_core.py)"]
        BC["Consistent Boundary Conditions (consistent_boundary_conditions.py)"]
    end

    %% Verbindungen
    Main_TwoMass --> SetupFuncs
    Main_NSP --> SetupFuncs
    Main_SCARA --> SetupFuncs
    Main_FiveBar --> SetupFuncs

    SetupFuncs --> ToolboxClass
    ToolboxClass --> OptimTask
    ToolboxClass --> FreeDyn
    ToolboxClass --> Control

    OptimTask --> UserFcts
    OptimTask --> AdjGrad
    AdjGrad --> BDF
    AdjGrad --> BC
    FreeDyn --> BDF
