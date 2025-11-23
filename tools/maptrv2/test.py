import pickle

file_path="/data/changxinyuan.cxy/NUSCENES/NUSCENES/nuscenes_map_infos_temporal_val_120m.pkl"
with open(file_path, 'rb') as f:
      map_infos = pickle.load(f)
import pdb; pdb.set_trace()