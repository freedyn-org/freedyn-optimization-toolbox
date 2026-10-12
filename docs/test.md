graph TD

&#x20;   %% Einstiegspunkte

&#x20;   subgraph Einstiegspunkte \["Hauptskripte (User Scripts)"]

&#x20;       Main\_TwoMass\[main\_TwoMass.py]\[cite: 1]

&#x20;       Main\_NSP\[main\_NSP.py]\[cite: 3]

&#x20;       Main\_SCARA\[main\_SCARA.py]\[cite: 4]

&#x20;       Main\_FiveBar\[main\_FiveBarMechanism.py]\[cite: 5]

&#x20;   end



&#x20;   %% Setup \& Toolbox

&#x20;   subgraph Setup \["Setup \& API Schicht (\_setup.py)"]

&#x20;       SetupFuncs\["OptimCtrl / TimeOptimCtrl / ParamIdent"]\[cite: 7]

&#x20;       ToolboxClass\["Toolbox (toolbox.py)"]\[cite: 9]

&#x20;   end



&#x20;   %% Kernkomponenten

&#x20;   subgraph Core \["Optimization Toolbox Core"]

&#x20;       OptimTask\["OptimTask (optimization\_task.py)"]\[cite: 8]

&#x20;       FreeDyn\["FreeDyn Management (Management\_FreeDyn.py)"]

&#x20;       Control\["Control / Splines (control\_cubSPL\_zeroClamped.py)"]

&#x20;       UserFcts\["User Functions (user\_fcts.py)"]\[cite: 2]

&#x20;   end



&#x20;   subgraph Solver \["Mathematischer Unterbau (Adjoint \& BDF)"]

&#x20;       AdjGrad\["Adjoint Gradients (adjGrad\_wrt\_u\_FDOP.py etc.)"]\[cite: 11, 12, 13]

&#x20;       BDF\["BDF Core \& Solvers (BDF\_core.py)"]\[cite: 14]

&#x20;       BC\["Consistent Boundary Conditions (consistent\_boundary\_conditions.py)"]\[cite: 17]

&#x20;   end



&#x20;   %% Verbindungen

&#x20;   Main\_TwoMass --> SetupFuncs

&#x20;   Main\_NSP --> SetupFuncs

&#x20;   Main\_SCARA --> SetupFuncs

&#x20;   Main\_FiveBar --> SetupFuncs



&#x20;   SetupFuncs --> ToolboxClass

&#x20;   ToolboxClass --> OptimTask

&#x20;   ToolboxClass --> FreeDyn

&#x20;   ToolboxClass --> Control



&#x20;   OptimTask --> UserFcts

&#x20;   OptimTask --> AdjGrad

&#x20;   AdjGrad --> BDF

&#x20;   AdjGrad --> BC

&#x20;   FreeDyn --> BDF

