# Execution script of the experiments for reproducibility

# The advecting pulse example
python experiments/pulse.py

# The Kuramoto-Sivashinsky equation example
python experiments/kse.py

# # AMR-Wind turbulent chnnel flow example
# # Noee that this does not run since the data is not 
# # publicly available
# python experiments/amr3Dchannel_u.py
# python experiments/amr3Dchannel_w.py
# python experiments/amr3Dchannel_viz.py

# Ablation study
# This takes a while so run it over night
python experiments/ablation.py