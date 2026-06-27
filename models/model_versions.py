from models.rbs import RBS


'''
    Base model with N=M=3
'''

def RBS_v1(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3)

'''
    Base model with N=M=5
'''

def RBS_v1_NM_5(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=5,
                  max_degree_n=5)

'''
    Same as RBS_v1, but we use damping
'''

def RBS_v2(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3,
                  use_damping_annealing=True)


'''
    Same as RBS_v1, but we also use offsets for the vibrating modes so that, hopefully, the gaps caused by the stitching effect will be covered. (test failed)
'''

def RBS_v3(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3,
                  predict_offset_for_each_mode=True)



'''
    Same as RBS_v1, but we use 2x upsampling
'''

def RBS_v4(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3,
                  upk=1)


'''
    Same as RBS_v1, but we use 4x upsampling
'''

def RBS_v5(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3,
                  upk=2)



'''
    Same as RBS_v5, but we use the alpha coverage loss function to reactivate dead branes. (test failed)
'''

def RBS_v6(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3,
                  upk=2,
                  use_alpha_coverage_loss = True)


'''
    Same as RBS_v5, but we scale the sigma based on the upk. 
'''

def RBS_v7(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3,
                  upk=2,
                  scale_sigma_based_on_upk = True)

'''
    Same as RBS_v7 but uses EDSR as the feature extractor
'''
def RBS_v8(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='edsr',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=3,
                  max_degree_n=3,
                  upk=2,
                  scale_sigma_based_on_upk = True)


'''
    Same as RBS_v7, but with N=M=5
'''
def RBS_v9(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=5,
                  max_degree_n=5,
                  upk=2,
                  scale_sigma_based_on_upk = True)


'''
    Same as RBS_v7, but with N=M=7
'''
def RBS_v10(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=7,
                  max_degree_n=7,
                  upk=2,
                  scale_sigma_based_on_upk = True)


'''
    Same as RBS_v7, but using degree N=M=2
'''
def RBS_v11(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=2,
                  max_degree_n=2,
                  upk=2,
                  scale_sigma_based_on_upk = True)



'''
    Same as RBS_v7, but I use degree N=M=0
'''
def RBS_v12(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=0,
                  max_degree_n=0,
                  upk=2,
                  scale_sigma_based_on_upk = True)

'''
    Just a test. I'm trying to adjust a lot of things, like the sigmoid threshold for the scale (but I haven't seen much of a change).
'''

def RBS_v13(**kwargs):
    return RBS(**kwargs, 
                  type_of_image_encoder='rdn',
                  default_step_size=1.0,
                  brane_color_type="per-mode",
                  max_degree_m=5,
                  max_degree_n=5,
                  relax_parameters = ["mean", "sigma"])



'''
    TOY EXAMPLE ------------
'''

def RBS_Checkerboard_Brane_7(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=7, M=7)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=7, 
                  max_degree_n=7,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,   
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 


def RBS_Checkerboard_Brane_6(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=6, M=6)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=6, 
                  max_degree_n=6,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,   
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 


def RBS_Checkerboard_Brane_5(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=5, M=5)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=5, 
                  max_degree_n=5,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,   
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 


def RBS_Checkerboard_Brane_4(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=4, M=4)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=4, 
                  max_degree_n=4,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,   
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 

def RBS_Checkerboard_Brane_3(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=3, M=3)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=3, 
                  max_degree_n=3,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,   
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 


def RBS_Checkerboard_Brane_2(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=2, M=2)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=2, 
                  max_degree_n=2,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,  
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 

def RBS_Checkerboard_Brane_1(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=1, M=1)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=1, 
                  max_degree_n=1,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,   
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 

def RBS_Checkerboard_Brane_0(**kwargs):
    """
    Proposed Model: Single Vibrating Branch (N=0, M=0)
    """
    return RBS(**kwargs, 
                  type_of_image_encoder='simple_cnn',
                  default_step_size=0.4,
                  brane_color_type="per-mode",
                  max_degree_m=0, 
                  max_degree_n=0,
                  dmax=2.0,
                  inchannel= 32,
                  channel= 32,
                  num_heads= 2,
                  num_crossattn_blocks= 1,
                  num_crossattn_layers= 1,
                  num_selfattn_blocks = 1,
                  num_selfattn_layers= 1,
                  num_gs_seed=1,  
                  window_size=12,  
                  use_alpha_coverage_loss=None,
                  freeze_mlp_mean=True) 


model_versions = {
    'RBS_v1': RBS_v1,
    'RBS_v1_NM_5': RBS_v1_NM_5,
    'RBS_v2': RBS_v2,
    'RBS_v3': RBS_v3,
    'RBS_v4': RBS_v4,
    'RBS_v5': RBS_v5,
    'RBS_v6': RBS_v6,
    'RBS_v7': RBS_v7,
    'RBS_v8': RBS_v8,
    'RBS_v9': RBS_v9,
    'RBS_v10': RBS_v10,
    'RBS_v11': RBS_v11,
    "RBS_v12": RBS_v12,
    "RBS_v13": RBS_v13,
    'RBS_checkboard_brane_7': RBS_Checkerboard_Brane_7,
    'RBS_checkboard_brane_6': RBS_Checkerboard_Brane_6,
    'RBS_checkboard_brane_5': RBS_Checkerboard_Brane_5,
    'RBS_checkboard_brane_4': RBS_Checkerboard_Brane_4,
    'RBS_checkboard_brane_3': RBS_Checkerboard_Brane_3,
    'RBS_checkboard_brane_2': RBS_Checkerboard_Brane_2,
    'RBS_checkboard_brane_1': RBS_Checkerboard_Brane_1,
    'RBS_checkboard_brane_0': RBS_Checkerboard_Brane_0,
}