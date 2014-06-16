PYTHON = python

# Train attribute network on CUHK_SAR
$(PYTHON) prepare_sar.py -d CUHK_SAR -i X S -t A
$(PYTHON) net_attr.py -d sar

# Train segmentation network on CUHK_SAR
$(PYTHON) prepare_sar.py -d CUHK_SAR -i X -t S -o seg
$(PYTHON) net_seg.py -d sar_seg -a attr

# Train latent network on Mix
$(PYTHON) prepare_attribute.py -d Mix
$(PYTHON) net_latent.py -d attribute -a attr -s seg

# Train segmentation network again but use latent network filters
$(PYTHON) net_seg.py -d sar_seg -a latent -o latent

# Run baseline methods on CUHK_SAR
$(PYTHON) baseline.py -d sar
$(PYTHON) net_attr.py -d sar --no-scpool -o no_scpool

$(PYTHON) evaluate.py -d sar -m baseline -o baseline
$(PYTHON) evaluate.py -d sar -m attr_no_scpool -o attr_no_scpool
$(PYTHON) evaluate.py -d sar -m attr -o attr

# Run baseline methods on Mix
$(PYTHON) baseline.py -d attribute -o mix
$(PYTHON) prepare_sar.py -d Mix_SAR -i X S -t A -o mix  # Need generate Mix_SAR by Matlab
$(PYTHON) net_attr.py -d sar_mix -o mix

$(PYTHON) evaluate.py -d attribute -m baseline_mix -o baseline_mix
$(PYTHON) evaluate.py -d attribute -m attr_mix -o attr_mix
$(PYTHON) evaluate.py -d attribute -m latent -o latent



