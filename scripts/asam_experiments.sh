# CIFAR-10
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="clean" --initial_rho=2.0 --adaptive=True --rho_scheduler="constant" --epochs=100
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="aggre" --initial_rho=2.0 --adaptive=True --rho_scheduler="constant" --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="worse" --initial_rho=2.0 --adaptive=True --rho_scheduler="constant" --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="blue" --noise=0.2 --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="blue" --noise=0.8 --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="blue" --noise=1.0 --epochs=200

# CIFAR-100
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="clean" --initial_rho=2.0 --adaptive=True --rho_scheduler="constant" --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="aggre" --initial_rho=2.0 --adaptive=True --rho_scheduler="constant" --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="worse" --initial_rho=2.0 --adaptive=True --rho_scheduler="constant" --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="blue" --noise=0.2 --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="blue" --noise=0.8 --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SAM" --label_type="blue" --noise=1.0 --epochs=200
