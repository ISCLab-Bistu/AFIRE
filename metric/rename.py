import os

# # 指定文件夹路径
# folder_path = 'E:/task/fusion\RFN-NEST\outputs/fused_rfnnest_700_wir_6.0_wvi_3.0_21_res\RFN-Nest_tno42_origin'
#
# # 遍历文件夹下的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.png'):
#         # 提取文件名中的数字部分
#         number = int(filename[:-4])
#
#         # 构造新的文件名
#         new_filename = f'IR{number}.png'
#
#         # 生成旧文件路径和新文件路径
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_filename)
#
#         # 重命名文件
#         os.rename(old_path, new_path)
#
# print('文件重命名完成。')


# def rename_images(directory):
#     for filename in os.listdir(directory):
#         if filename.startswith("VIS") and filename.endswith(".png"):
#             new_filename = filename.replace("VIS", "IR")
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
#
# # 指定包含图片的文件夹路径
# image_directory = "E:/task/fusion\SKFusion\images\M3FD/vi"
#
# # 调用函数来重命名图片
# rename_images(image_directory)


# def rename_images(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".png"):
#             parts = filename.split(".")
#             basename = parts[0]
#             new_basename = "IR" + str(int(basename))
#             new_filename = new_basename + "." + parts[1]
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
#
# # 指定包含图片的文件夹路径
# image_directory = "E:/task/fusion\SKFusion\images\M3FD/ir"
#
# # 调用函数来重命名图片
# rename_images(image_directory)



#
# def rename_images(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".png"):
#             basename = os.path.splitext(filename)[0]
#             new_basename = "VIS" + basename.lstrip("0")
#             new_filename = new_basename + ".png"
#             os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
#
# # 指定包含图片的子文件夹路径
# image_directory = "E:/task/fusion/SKFusion/images/LLVIP/vis"
#
# # 调用函数来重命名图片
# rename_images(image_directory)


import os


def add_prefix_to_files(folder_path, prefix):
    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return

    file_list = os.listdir(folder_path)

    for file_name in file_list:
        _, extension = os.path.splitext(file_name)  # 获取文件名和扩展名
        new_file_name = f"{prefix}{file_name}"

        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_file_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_file_name}")


if __name__ == "__main__":
    folder_path = r"E:\task\fusion\SKFusion\output_image_tno21\U2Fusion_tno21"  # 替换为实际的文件夹路径
    prefix = "IR"
    add_prefix_to_files(folder_path, prefix)

