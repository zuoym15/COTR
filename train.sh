python train_cater.py --use_ram=yes --batch_size=8 \
--learning_rate=1e-4 --lr_backbone=1e-5 --max_iter=600000 --workers=8 \
--cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --confirm=no \
--suffix=stage_1 --valid_iter=20000 --enable_zoom=no \
#--#resume=yes \
