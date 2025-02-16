# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#define your pre-trained (post-trained) models here with their paths.

dependency_folder = "../../../ERT-for-RRC-ABSA_dependent_files"

MODEL_ARCHIVE_MAP = {
    'bert-base': 'bert-base-uncase',
    'bert-large': 'bert-large-uncase',

    'laptop_pt_review': f'{dependency_folder}/pt_model/laptop_pt_review/',
    'rest_pt_review': f'{dependency_folder}/pt_model/rest_pt_review/',
    
    'pt_squad': f'{dependency_folder}/pt_model/pt_squad/',
    
    'laptop_pt': f'{dependency_folder}/pt_model/laptop_pt/',
    'rest_pt': f'{dependency_folder}/pt_model/rest_pt/',  
}
