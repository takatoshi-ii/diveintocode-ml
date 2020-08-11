from predictor import ScoringService
from os import listdir
import json

ScoringService.get_model()

input_folder = r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\video"
output_folder = r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\annotation"
video_file_names = [f for f in listdir("train_videos") if f.endswith(".mp4")]

print(video_file_names)

for video_file_name in video_file_names:
    full_video_file_name = "/".join([input_folder, video_file_name])
    full_json_file_name = "/".join([output_folder, video_file_name + ".json"])
    print("{} into {}".format(full_video_file_name, full_json_file_name))
    print("\tpredicting")
    res = ScoringService.predict(full_video_file_name)
    print("\tsaving")
    with open(full_json_file_name, "w") as f:
        json.dump(res, f)
