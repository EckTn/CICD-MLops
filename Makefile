install:
	pip	install --upgrade pip &&\
    pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
#	echo "![Confusion Matrix](./results/confusion_matrix.png)" >> report.md

	# Debbuging
	if [ -f "./results/confusion_matrix.png"]; then \
  		echo "![Confusion Matrix](./results/confusion_matrix.png)" >> report.md; \
  	else \
  	  	echo "Image not found" >> report.md; \
  	fi

	cat report.md
	ls -lh report.md

	cml comment create report.md