import torch
import torch.nn as nn
import math
from einops import rearrange
from models.attention.attention import WindowCrossAttnBlock, GSSelfAttnBlock
#from utils.plots import calculate_complexity_map, plot_gt_vs_complexity


class RBS(nn.Module):
    def __init__(self, 
                    inchannel=64, 
                    channel=180, 
                    num_heads=6, 
                    num_crossattn_blocks=1, 
                    num_crossattn_layers=2, 
                    num_selfattn_blocks = 6, 
                    num_selfattn_layers = 6,
                    num_gs_seed=144, 
                    gs_up_factor=1.0, 
                    window_size=12, 
                    use_checkpoint = False,
                    type_of_image_encoder = 'rdn',
                    default_step_size = 1.2,
                    
                    #RBS
                    brane_color_type = "per-mode", # [“constant”] #If set to “constant,” the brana will have a base color; if set to “per-mode,” each vibrating mode will have its own color
                    max_degree_n = 2, # N_max
                    max_degree_m = 2, # M_max
                    dmax = 0.1,
                    use_damping_annealing = False, #When using dumping, the polynomial coefficients will have less and less influence on the color as the degree increases
                    predict_offset_for_each_mode = False,
                    upk = None, #If up = 2*k, then it will perform 2x upsampling of the feature map before decoding
                    use_alpha_coverage_loss = None,
                    scale_sigma_based_on_upk = False,

                    freeze_mlp_mean = False,
                    relax_parameters = None
                    ):

        super(RBS, self).__init__()
        self.channel = channel
        self.nhead = num_heads
        self.gs_up_factor = gs_up_factor
        self.num_gs_seed = num_gs_seed
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.type_of_image_encoder = type_of_image_encoder
        self.brane_color_type = brane_color_type
        self.default_step_size = default_step_size
        self.max_degree_n = max_degree_n
        self.max_degree_m = max_degree_m
        self.dmax = dmax
        self.use_damping_annealing = use_damping_annealing
        self.predict_offset_for_each_mode = predict_offset_for_each_mode
        self.upk = upk
        self.use_alpha_coverage_loss = use_alpha_coverage_loss
        self.scale_sigma_based_on_upk = scale_sigma_based_on_upk
        self.relax_parameters = relax_parameters

        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))
        self.gs_up_factor_sqrt = int(math.sqrt(gs_up_factor))

        # shared gaussian embedding and its pos embedding
        self.gs_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)

        if type_of_image_encoder == 'rdn':
            from models.image_encoder.rdn.rdn import RDNNOUP
            self.image_encoder = RDNNOUP()
        elif type_of_image_encoder == 'edsr':
            from models.image_encoder.edsr.edsr import EDSRNOUP
            self.image_encoder = EDSRNOUP()
        elif type_of_image_encoder == 'simple_cnn':
            self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, inchannel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1)
                )
        else:
            raise Exception(f"The type of image encoder [{self.image_encoder}] is not evailable.")

        self.img_feat_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )

        self.window_crossattn_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_layers,
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_blocks)
        ])

        self.gs_selfattn_blocks = nn.ModuleList([
            GSSelfAttnBlock(dim=channel,
                            num_heads=num_heads,
                            num_selfattn_layers=num_selfattn_layers,
                            num_gs_seed_sqrt=self.num_gs_seed_sqrt
                            ) for i in range(num_selfattn_blocks)
        ])

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, channel)
        )


        # Brane Head Predictors
        self.mlp_sigma = self._build_mlp(channel, 2)   # Sx, Sy
        self.mlp_theta = self._build_mlp(channel, 1)   # Rotation angles
        self.mlp_alpha = self._build_mlp(channel, 1)   # alpha brana
        self.mlp_mean  = self._build_mlp(channel, 2)   # offsets x0, y0

        torch.nn.init.zeros_(self.mlp_mean[-1].weight)
        torch.nn.init.zeros_(self.mlp_mean[-1].bias)

        if freeze_mlp_mean == True:
            last_linear = self.mlp_mean[-1]
    
            torch.nn.init.zeros_(last_linear.weight)
            torch.nn.init.zeros_(last_linear.bias)
            
            for param in self.mlp_mean.parameters():
                param.requires_grad = False

        
        self.num_modi = (max_degree_n + 1) * (max_degree_m + 1)

        if self.brane_color_type == "per-mode":
            # An RGB vector for each vibrating mode
            self.mlp_rgb_modes = self._build_mlp(channel, self.num_modi * 3)

            if self.predict_offset_for_each_mode: 
                #Predict (dx, dy) for each vibrating mode
                self.mlp_mode_offsets = self._build_mlp(channel, self.num_modi * 2)
            
            if self.use_damping_annealing:
                #damping list
                self.damping_list = []
                for n in range(max_degree_n + 1):
                    for m in range(max_degree_m + 1):
                        if n == 0 and m == 0:
                            continue 
                        
                        # Decay: The higher modes gradually carry less weight
                        weight = 1.0 / (n + m + 1.0)
                        self.damping_list.extend([weight, weight, weight])
            else:
                self.damping_list = None      

        elif self.brane_color_type == "constant":
            # An RGB base color + a weight for each mode
            self.mlp_rgb_base = self._build_mlp(channel, 3)
            self.mlp_mode_coeffs = self._build_mlp(channel, self.num_modi)
        

        if self.upk is not None:
            layers = []
            for _ in range(self.upk):
                layers.append(nn.Conv2d(channel, channel * 4, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
                
            self.UPNet = nn.Sequential(*layers)
            self.total_scale = 2 ** self.upk
        else:
            self.total_scale = 1


    def _build_mlp(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch * 4),
            nn.ReLU(),
            nn.Linear(in_ch * 4, out_ch)
        )



    @staticmethod
    def get_N_reference_points(h, w, device='cuda'):
        step_y = 1 / h
        step_x = 1 / w
        ref_y, ref_x = torch.meshgrid(torch.linspace(step_y / 2, 1 - step_y / 2, h, dtype=torch.float32, device=device),
                                      torch.linspace(step_x / 2, 1 - step_x / 2, w, dtype=torch.float32, device=device))
        reference_points = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = reference_points[None, :, None]
        return reference_points


    def forward(self, lr_image, scale, inference=False, grt_image=None):
        device = lr_image.device
        srcs = self.image_encoder(lr_image)  # b,c,h,w
        b, c, h, w = srcs.shape
        scale = torch.full((b, 1), float(scale), device=device)

        b, c, h, w = srcs.shape  ###srcs is pad to the size that could be divided by window_size
        query = self.gs_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (w // self.window_size),
                                                                   1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query = query.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                              self.channel)  # b*h_count*w_count, num_gs_seed, channel


        scale = 1 / scale
        #scale = scale.unsqueeze(1)  # b*1
        scale_embedding = self.scale_mlp(scale)  # b*channel
        scale_embedding = scale_embedding.unsqueeze(1).unsqueeze(2).repeat(1, (h // self.window_size) * (
                    w // self.window_size), self.num_gs_seed, 1)  # b, h_count*w_count, num_gs_seed, channel
        scale_embedding = scale_embedding.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel) # b*h_count*w_count, num_gs_seed, channel

        query_pos = self.pos_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                    w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel

        feat = self.img_feat_proj(srcs)  # b*channel*h*w

        query_pos = query_pos.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel)  # b*h_count*w_count, num_gs_seed, channel

        for block in self.window_crossattn_blocks:
            query = block(query, query_pos, feat, scale_embedding)  # b*h_count*w_count, num_gs_seed, channel

        resi = query
        for block in self.gs_selfattn_blocks:
            query = block(query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
        query = query + resi

        query = rearrange(query, '(b m n) (h w) c -> b c (m h) (n w)', m=h // self.window_size, n=w // self.window_size,
                          h=self.num_gs_seed_sqrt)


        if self.upk is not None:
            query = self.UPNet(query)
            
        query = query.permute(0,2,3,1)

        # extract geometric parameters
        query_sigma = self.mlp_sigma(query).reshape(b, -1, 2)
        query_theta = self.mlp_theta(query).reshape(b, -1, 1)
        query_alpha = self.mlp_alpha(query).reshape(b, -1, 1)
        query_mean  = self.mlp_mean(query).reshape(b, -1, 2)

        # 2. Calculate the final dimensions for width and height
        scaled_w = self.num_gs_seed_sqrt * (w // self.window_size) * self.total_scale
        scaled_h = self.num_gs_seed_sqrt * (h // self.window_size) * self.total_scale

        scale_tensor = torch.tensor([scaled_w, scaled_h], device=query_mean.device, dtype=query_mean.dtype)
        
        if not (self.relax_parameters is not None and "mean" in self.relax_parameters):
            query_mean = query_mean / scale_tensor  

        # 3. Compute reference position of each Brane
        reference_offset = self.get_N_reference_points(scaled_h, scaled_w, srcs.device)



        # 4. Add predicted offset to the reference position of each Brane
        query_mean = query_mean + reference_offset.reshape(1, -1, 2)

        if self.predict_offset_for_each_mode: 
            query_offsets = self.mlp_mode_offsets(query).reshape(b, -1, self.num_modi, 2)

            max_shift_ndc = 2.0 / torch.tensor(
                [self.num_gs_seed_sqrt * (w // self.window_size), 
                 self.num_gs_seed_sqrt * (h // self.window_size)], 
                dtype=torch.float32, device=query_offsets.device)[None, None, None, :]
                
            query_offsets = torch.tanh(query_offsets) * max_shift_ndc
            
            query_offsets = query_offsets.reshape(b, -1, self.num_modi * 2)


        if self.brane_color_type == "per-mode":
            query_colors = self.mlp_rgb_modes(query).reshape(b, -1, self.num_modi * 3)
            
            if self.predict_offset_for_each_mode:
                out = torch.cat([query_sigma, query_theta, query_alpha, query_colors, query_offsets, query_mean], dim=-1)
            else:
                # [Sx, Sy, Theta, Alpha, R0,G0,B0, ... Rn,Gn,Bn, X, Y]
                out = torch.cat([query_sigma, query_theta, query_alpha, query_colors, query_mean], dim=-1)
            
        elif self.brane_color_type == "constant":
            query_rgb_base = self.mlp_rgb_base(query).reshape(b, -1, 3)
            query_coeffs = self.mlp_mode_coeffs(query).reshape(b, -1, self.num_modi)
            # [Sx, Sy, Theta, Alpha, Rbase,Gbase,Bbase, c0, c1, ... cn, X, Y]
            out = torch.cat([query_sigma, query_theta, query_alpha, query_rgb_base, query_coeffs, query_mean], dim=-1)




        '''
            PLOTTING
        '''
        # # Extract complexity map
        # complexity_map = calculate_complexity_map(
        #     out_tensor=out, 
        #     h_out=scaled_h, 
        #     w_out=scaled_w, 
        #     num_modi=self.num_modi, 
        #     brane_color_type=self.brane_color_type,
        #     predict_offset_for_each_mode=self.predict_offset_for_each_mode
        # )

        # # Visualize or save complexity map
        # plot_gt_vs_complexity(grt_image, complexity_map, batch_idx=5, save_path="./da_eliminare/plots/complexity_ablation.png")




        return out
