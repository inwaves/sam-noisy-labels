# CIFAR-10
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="clean" --initial_rho=0.5 --adaptive=False --rho_scheduler="constant" --epochs=100
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="aggre" --initial_rho=0.5 --adaptive=False --rho_scheduler="constant" --epochs=100
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="worse" --initial_rho=0.5 --adaptive=False --rho_scheduler="constant" --epochs=100
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="blue" --noise=0.2 --epochs=100
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="blue" --noise=0.8 --epochs=100
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="blue" --noise=1.0 --epochs=100

# CIFAR-100
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="clean" --initial_rho=0.5 --adaptive=False --rho_scheduler="constant" --epochs=100
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="aggre" --initial_rho=0.5 --adaptive=False --rho_scheduler="constant" --epochs=100
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="worse" --initial_rho=0.5 --adaptive=False --rho_scheduler="constant" --epochs=100
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="blue" --noise=0.2 --epochs=100
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="blue" --noise=0.8 --epochs=100
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="blue" --noise=1.0 --epochs=100
