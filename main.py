import os
import argparse

if __name__ == '__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument('-m',dest='method',default='SneakyPrompt',
                        choices=['Ring-A-Bell', "SneakyPrompt", "UnlearnDiff", "MMA_Diff"],
                        help='The attack method you want to use')
    parser.add_argument('-g',dest='gpu',default='7')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from baseline_attack import attack_process
    attack_process(args)