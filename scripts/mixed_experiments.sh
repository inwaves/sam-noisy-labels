python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.5 --adaptive=False --label_type="worse" --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.5 --adaptive=False --label_type="blue" --noise=0.4 --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.5 --rho_scheduler="stepdecay" --adaptive=False --label_type="worse"  --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.5 --rho_scheduler="stepdecay" --adaptive=False --label_type="blue" --noise=0.4 --epochs=200

python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --adaptive=True --label_type="worse" --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --adaptive=True --label_type="blue" --noise=0.4 --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --rho_scheduler="stepdecay" --adaptive=True --label_type="worse"  --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --rho_scheduler="stepdecay" --adaptive=True --label_type="blue" --noise=0.4 --epochs=200