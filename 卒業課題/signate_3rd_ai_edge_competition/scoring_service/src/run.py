from predictor import ScoringService
from os import listdir
import json

print("get_model")
ScoringService.get_model(r"D:\SIGNATE\Signate_3rd_AI_edge_competition\model")

input_folder = r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\video"
output_folder = r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\annotation"
video_file_names = [f for f in listdir(r"D:\SIGNATE\Signate_3rd_AI_edge_competition\test\video") if f.endswith(".mp4")]

print(video_file_names)

for video_file_name in video_file_names:
    full_video_file_name = "/".join([input_folder, video_file_name])
    full_json_file_name = "/".join([output_folder, video_file_name + ".json"])
    print("{} into {}".format(full_video_file_name, full_json_file_name))
    #print("\t ★★★★★predicting")

    print("predict")
    res = ScoringService.predict(full_video_file_name)
    print("\t ★★★★★saving")
    with open(full_json_file_name, "w") as f:
        json.dump(res, f)
