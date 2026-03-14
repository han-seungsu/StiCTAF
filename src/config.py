import argparse

def get_args(arg_list=None):

    parser = argparse.ArgumentParser(
        description="Train normalizing flow model with KLD loss"
    )
    parser.add_argument('--convert_to_t', type=bool, default=False)
    parser.add_argument('--add_ttf', type=bool, default=False)
    parser.add_argument('--add_StudentTttf', type=bool, default=False)
    parser.add_argument('--ttf_init_scale', type=float, default=2.0)
    parser.add_argument('--ttf_threshold', type=float, default=1e-2)
    parser.add_argument('--tail_nsamples', type=int, default=5000)
    parser.add_argument('--freeze_flow', default =True)
    parser.add_argument('--freeze_ratio', type=float, default=0.8)
    parser.add_argument('--base_lr',   type=float, default=5e-3)
    parser.add_argument('--flow_lr',   type=float, default=5e-3)
    parser.add_argument('--weight_lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_iter',    type=int,   default=500)
    parser.add_argument('--num_samples', type=int,   default=2**10)
    parser.add_argument(
        '--loss_type',
        choices=['forward','reverse','stratified','siw','componentwise'],
        default='reverse',
    )
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help="Save the model after training")
    parser.add_argument('--save_path', type=str, default='/home/gkstmtm/2025_NF')
    parser.add_argument('--file_name', type=str, default='model')
    
    return parser.parse_args(arg_list)