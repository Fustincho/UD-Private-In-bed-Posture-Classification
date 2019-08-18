# These functions are introduced along the Part 1 notebook.

# Position vectors. We load the data with respect to the file name, which is
# a number corresponding to a specific in-bed position. We take advantage of this
# and use the number to get the position with help of the following vectors.

positions_i = ["justAPlaceholder", "supine", "right",
               "left", "right", "right",
               "left", "left", "supine",
               "supine", "supine", "supine",
               "supine", "right", "left",
               "supine", "supine", "supine"]

positions_ii = {
    "B":"supine", "1":"supine", "C":"right",
    "D":"left", "E1":"right", "E2":"right",
    "E3":"left", "E4":"left", "E5":"right",
    "E6":"left", "F":"supine", "G1":"supine",
    "G2":"right", "G3":"left"
}

# We also want the classes to be encoded as numbers so we can work easier when
# modeling. This function achieves so. Since left_fetus and right_fetus are not
# considered as classes in the evaluation of the original paper and since they
# are not considered in the "Experiment I", we encode them also as left and right
# positions.

def token_position(x):
  return {
      'supine': 0,
      'left': 1,
      'right': 2,
      'left_fetus': 1,
      'right_fetus': 2
  }[x]

def load_exp_i(path):
  """
  Creates a numpy array for the data and labels.

  params:
  ------
  path    -- Data path.

  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        for file in files:
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            for line in f.read().splitlines()[2:]:

              raw_data = np.fromstring(line, dtype=float, sep='\t')
              # Change the range from [0-1000] to [0-255].
              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              Normalize = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.ToTensor()
                          ])
              file_data = Normalize(file_data.reshape(64,32))
              file_data = file_data.view(1, 64, 32)
              # Turn the file index into position list,
              # and turn position list into reduced indices.
              file_label = token_position(positions_i[int(file[:-4])])
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (torch.from_numpy(data), torch.from_numpy(labels))
  return dataset

# Both air and sponge mattresses used in the data collection have a different
# size (64 x 27), opposed to the pressure mattress (64 x 32) used in the first
# experiment. Additionally, the image is rotated by 180 degrees with respect to
# the experiment one images.
# This function serves to set the format of the images equal to the ones taken
# by the pressure mat.

def resize_and_rotate(image):
  To_PIL_and_Resize = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.Resize((64, 32))
                      ])

  rotated = TF.rotate(To_PIL_and_Resize(image), angle=180)

  return transforms.ToTensor()(rotated)


def load_exp_ii(path):

  exp_ii_data_air = {}
  exp_ii_data_spo = {}

  # each directory is a subject
  for _, subject_dirs, _ in os.walk(path):
    for subject in subject_dirs:
      data = None
      labels = None

      # each directory is a matresss
      for _, mat_dirs, _ in os.walk(os.path.join(path, subject)):
        for mat in mat_dirs:
          for _, _, files in os.walk(os.path.join(path, subject, mat)):
            for file in files:
              file_path = os.path.join(path, subject, mat, file)
              raw_data = np.loadtxt(file_path)
              # Change the range from [0-500] to [0-255].
              file_data = np.round(raw_data*255/500).astype(np.uint8)
              file_data = resize_and_rotate(file_data)
              file_data = file_data.view(1, 64, 32)

              if file[-6] == "E" or file[-6] == "G":
                file_label = positions_ii[file[-6:-4]]
              else:
                file_label = positions_ii[file[-6]]

              file_label = token_position(file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)

              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

          if mat == "Air_Mat":
            exp_ii_data_air[subject] = (torch.from_numpy(data), torch.from_numpy(labels))
          else:
            exp_ii_data_spo[subject] = (torch.from_numpy(data), torch.from_numpy(labels))

          data = None
          labels = None

    return exp_ii_data_air, exp_ii_data_spo
