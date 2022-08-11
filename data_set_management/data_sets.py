import os


class DapiDataSet1:
    def __init__(self, data_set_dir, **kwargs):
        super().__init__(**kwargs)  # forwards all unused arguments
        self.data_set_dir = data_set_dir

    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list({file.split("_c")[0] for file in files})

    def get_dapi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch4.tiff")

    def get_cell_image_path(self, file):
        raise ValueError("Cell segmentation is not available in this dataset")

    def get_bright_field_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch5.tiff")

    def get_phase_contrast_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch1.tiff")

    def get_fucci_golgi_image_path(self, file):
        return os.path.join(self.data_set_dir, f"{file}_ch3.tiff")

    def get_nucleus_semantic_image_path(self, file):
        # Must be updated
        return os.path.join(self.data_set_dir, file + "_cellpose.png")


class DapiDataSet2:
    def __init__(self, data_set_dir, **kwargs):
        super().__init__(**kwargs)  # forwards all unused arguments
        self.data_set_dir = data_set_dir

    def get_distinct_files(self):
        files = os.listdir(self.data_set_dir)
        return list(
            {file.split("_")[0] + " " + file.split("_")[-1].split(".")[0] for file in files}
        )

    def get_dapi_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_w3Hoechst_{splitted_file[1]}.TIF"
        )

    def get_cell_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(self.data_set_dir, f"{splitted_file[0]}_w4Cy5_{splitted_file[1]}.TIF")

    def get_dic_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_w5DIC-oil-40x_{splitted_file[1]}.TIF"
        )

    def get_cell_semantic_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_cell_semantic_{splitted_file[1]}.png"
        )

    def get_cell_topology_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_cell_topology_{splitted_file[1]}.png"
        )

    def get_cell_instance_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_cell_instance_{splitted_file[1]}.png"
        )

    def get_nucleus_instance_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_nucleus_instance_{splitted_file[1]}.png"
        )

    def get_nucleus_semantic_image_path(self, file):
        splitted_file = file.split(" ")
        return os.path.join(
            self.data_set_dir, f"{splitted_file[0]}_nucleus_semantic_{splitted_file[1]}.png"
        )
